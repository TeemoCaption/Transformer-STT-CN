import os
from datasets import load_from_disk
from src.data_utils import load_commonvoice_datasets, merge_datasets, preprocess_dataset, create_and_save_vocab
from src.utils import get_processor, prepare_batch, frame_generator, vad_collector

import tensorflow as tf
import numpy as np
from transformers import TFWav2Vec2ForCTC
import editdistance  # 用於計算 CER
import webrtcvad  # 用於靜音過濾

# 啟用 mixed precision
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# 設定 GPU 記憶體動態配置
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

def compute_cer(pred_str, ref_str):
    """
    計算 CER = (字級 Levenshtein 距離) / (參考句子長度)
    """
    distance = editdistance.eval(pred_str, ref_str)
    return distance / len(ref_str) if len(ref_str) > 0 else 0.0

class EvaluateCERCallback(tf.keras.callbacks.Callback):
    """
    在每個 epoch 結束後，用驗證集計算 CER。
    """
    def __init__(self, valid_dataset, processor):
        super().__init__()
        self.valid_dataset = valid_dataset
        self.processor = processor

    @tf.function(reduce_retracing=True)
    def predict_batch(self, x, y):
        # 封裝模型預測，以降低 retracing 次數
        return self.model.hf_model(x, labels=y, training=False)

    def on_epoch_end(self, epoch, logs=None):
        total_cer = 0.0
        count = 0
        for x, y in self.valid_dataset:
            outputs = self.predict_batch(x, y)
            predicted_ids = tf.argmax(outputs.logits, axis=-1)
            predicted_strs = self.processor.tokenizer.batch_decode(
                predicted_ids.numpy(), skip_special_tokens=True
            )
            # 使用 tf.tensor_scatter_nd_update 替換 -100 為 pad_token_id
            mask = tf.equal(y, -100)
            indices = tf.where(mask)
            updates = tf.fill([tf.shape(indices)[0]], self.processor.tokenizer.pad_token_id)
            ground_truth_ids = tf.tensor_scatter_nd_update(y, indices, updates)
            ground_truth_strs = self.processor.tokenizer.batch_decode(
                ground_truth_ids.numpy(), skip_special_tokens=True
            )
            for pred, ref in zip(predicted_strs, ground_truth_strs):
                total_cer += compute_cer(pred, ref)
                count += 1

        avg_cer = total_cer / count if count > 0 else 0.0
        print(f"Validation CER: {avg_cer:.4f}")
        if logs is not None:
            logs["val_cer"] = avg_cer

class KerasWav2Vec2ForCTC(tf.keras.Model):
    """
    將 Hugging Face 的 TFWav2Vec2ForCTC 包裝成一個可用 Keras model.fit() 的模型，
    並在 train_step/test_step 中使用 Hugging Face 模型計算 loss。
    """
    def __init__(self, hf_model):
        super().__init__()
        self.hf_model = hf_model

    def train_step(self, data):
        x, y = data
        max_label = tf.reduce_max(y)
        if max_label >= self.hf_model.config.vocab_size:
            print(f"Batch max label ID: {max_label}, Invalid labels in batch: {y}")
        with tf.GradientTape() as tape:
            outputs = self.hf_model(x, labels=y, training=True)
            loss = outputs.loss
        train_vars = self.trainable_variables
        grads = tape.gradient(loss, train_vars)
        self.optimizer.apply_gradients(zip(grads, train_vars))
        return {"loss": loss}

    def test_step(self, data):
        x, y = data
        outputs = self.hf_model(x, labels=y, training=False)
        loss = outputs.loss
        return {"loss": loss}

def webrtcvad_in_memory(audio_array, sample_rate=16000, frame_duration_ms=30, padding_duration_ms=300, aggressiveness=3):
    """
    使用 webrtcvad 去除靜音。
    """
    int16audio = (audio_array * 32767).astype(np.int16)
    raw_pcm = int16audio.tobytes()
    vad = webrtcvad.Vad(aggressiveness)
    frames = list(frame_generator(frame_duration_ms, raw_pcm, sample_rate))
    segments = vad_collector(sample_rate, frame_duration_ms, padding_duration_ms, vad, frames)
    voiced_pcm = b''.join([seg[1] for seg in segments])
    if len(voiced_pcm) == 0:
        return np.zeros(0, dtype=np.float32)
    new_int16 = np.frombuffer(voiced_pcm, dtype=np.int16)
    new_float = new_int16.astype(np.float32) / 32767.0
    return new_float

def main():
    preprocessed_path = "dataset/preprocessed"
    if os.path.exists(preprocessed_path):
        train_dataset = load_from_disk(os.path.join(preprocessed_path, "train"))
        valid_dataset = load_from_disk(os.path.join(preprocessed_path, "valid"))
        test_dataset  = load_from_disk(os.path.join(preprocessed_path, "test"))
        print("載入已快取的資料集（包含前一次 VAD 結果）")
    else:
        (cv_zh_train, cv_zh_valid, cv_zh_test,
         cv_tai_train, cv_tai_valid, cv_tai_test) = load_commonvoice_datasets()
        train_dataset = merge_datasets(cv_zh_train, cv_tai_train, split_name="train")
        valid_dataset = merge_datasets(cv_zh_valid, cv_tai_valid, split_name="valid")
        test_dataset  = merge_datasets(cv_zh_test, cv_tai_test, split_name="test")
        train_dataset, valid_dataset, test_dataset = preprocess_dataset(
            train_dataset, valid_dataset, test_dataset
        )
        def apply_webrtcvad(example):
            sr = example["audio"]["sampling_rate"]
            if sr != 16000:
                # 如有需要，可先重採樣到 16k
                pass
            float_array = example["audio"]["array"]
            new_array = webrtcvad_in_memory(
                audio_array=float_array,
                sample_rate=sr,
                frame_duration_ms=30,
                padding_duration_ms=300,
                aggressiveness=3
            )
            example["audio"]["array"] = new_array
            return example

        train_dataset = train_dataset.map(apply_webrtcvad)
        valid_dataset = valid_dataset.map(apply_webrtcvad)
        test_dataset  = test_dataset.map(apply_webrtcvad)
        os.makedirs(os.path.join(preprocessed_path, "train"), exist_ok=True)
        os.makedirs(os.path.join(preprocessed_path, "valid"), exist_ok=True)
        os.makedirs(os.path.join(preprocessed_path, "test"), exist_ok=True)
        train_dataset.save_to_disk(os.path.join(preprocessed_path, "train"))
        valid_dataset.save_to_disk(os.path.join(preprocessed_path, "valid"))
        test_dataset.save_to_disk(os.path.join(preprocessed_path, "test"))
        print("VAD 處理完畢，並已將資料集存到", preprocessed_path)

    train_sample = train_dataset[0]
    print("第一筆訓練資料:")
    print("取樣率:", train_sample["audio"]["sampling_rate"])
    print("音訊長度:", len(train_sample["audio"]["array"]))
    print("句子:", train_sample["sentence"])

    tokenizer, vocab_dict = create_and_save_vocab(train_dataset)
    processor = get_processor(tokenizer)

    train_dataset = train_dataset.map(lambda batch: prepare_batch(batch, processor),
                                      remove_columns=train_dataset.column_names)
    valid_dataset = valid_dataset.map(lambda batch: prepare_batch(batch, processor),
                                      remove_columns=valid_dataset.column_names)
    test_dataset  = test_dataset.map(lambda batch: prepare_batch(batch, processor),
                                     remove_columns=test_dataset.column_names)

    print(train_dataset.features)

    batch_size = 1
    def train_generator():
        for sample in train_dataset:
            yield sample["input_values"], sample["labels"]
    output_signature = (
        tf.TensorSpec(shape=(None,), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.int32),
    )
    train_tfds = tf.data.Dataset.from_generator(train_generator, output_signature=output_signature)
    train_tfds = train_tfds.shuffle(train_dataset.num_rows)
    train_tfds = train_tfds.padded_batch(
        batch_size=batch_size,
        padded_shapes=([None], [None]),
        padding_values=(0.0, -100)
    )
    def valid_generator():
        for sample in valid_dataset:
            yield sample["input_values"], sample["labels"]
    valid_tfds = tf.data.Dataset.from_generator(valid_generator, output_signature=output_signature)
    valid_tfds = valid_tfds.padded_batch(
        batch_size=batch_size,
        padded_shapes=([None], [None]),
        padding_values=(0.0, -100)
    )

    # 載入預訓練模型，注意使用 from_pt=True 從 PyTorch 權重轉換，但在 mixed precision 下，這是正常現象
    pretrained_model_name = "facebook/wav2vec2-base"

    # 暫時設置全域策略為 float32 以載入模型
    tf.keras.mixed_precision.set_global_policy('float32')

    hf_model = TFWav2Vec2ForCTC.from_pretrained(
        pretrained_model_name,
        vocab_size=len(vocab_dict),
        pad_token_id=tokenizer.pad_token_id,
        from_pt=True
    )

    print("Tokenizer vocab size:", processor.tokenizer.vocab_size)
    print("Model config vocab size:", hf_model.config.vocab_size)

    # 設置回 mixed_float16 以進行訓練
    tf.keras.mixed_precision.set_global_policy('mixed_float16')

    # 凍結特徵萃取 CNN 層
    hf_model.wav2vec2.feature_extractor.trainable = False
    # 凍結除了最後 N 層外的所有 Transformer 層（僅訓練最後 3 層）
    N = 3
    for layer in hf_model.wav2vec2.encoder.layer[:-N]:
        layer.trainable = False

    # 包裝成 Keras 模型
    model = KerasWav2Vec2ForCTC(hf_model)
    trainable_params = np.sum([np.prod(var.shape) for var in model.trainable_variables])
    total_params = np.sum([np.prod(var.shape) for var in model.variables])
    print(f"可訓練參數/總參數: {trainable_params} / {total_params}")

    # 使用混合精度的 LossScaleOptimizer
    base_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    optimizer = tf.keras.mixed_precision.LossScaleOptimizer(base_optimizer)
    model.compile(optimizer=optimizer, run_eagerly=False)

    cer_callback = EvaluateCERCallback(valid_tfds, processor)
    model.fit(
        train_tfds,
        epochs=3,
        callbacks=[cer_callback]
    )

if __name__ == "__main__":
    main()