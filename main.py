import os
import json
from datasets import load_from_disk
from tqdm.auto import tqdm
from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    TFWav2Vec2ForCTC
)
import tensorflow as tf
import numpy as np
import editdistance

from src.data_utils import (
    load_commonvoice_datasets,
    merge_datasets,
    preprocess_dataset,
    create_and_save_vocab,
    debug_check_labels,
    filter_invalid_chars
)
from src.utils import get_processor, prepare_batch

tf.get_logger().setLevel('ERROR')

# 設定 GPU 記憶體動態配置
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

class KerasWav2Vec2ForCTC(tf.keras.Model):
    def __init__(self, hf_model, processor):
        super().__init__()
        self.hf_model = hf_model
        self.processor = processor

    def call(self, inputs, training=False):
        # 如果 inputs 為一組 tensor，直接呼叫 hf_model
        return self.hf_model(inputs, training=training)
    
    def train_step(self, data):
        x, y = data

        y = tf.where(y < 0, 0, y)
        if tf.reduce_max(y).numpy() >= self.hf_model.config.vocab_size:
            raise ValueError(f"標籤值必須 <= vocab_size: {self.hf_model.config.vocab_size}")

        with tf.GradientTape() as tape:
            outputs = self.hf_model(x, labels=y, training=True)
            loss = outputs.loss

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return {"loss": loss}

    def test_step(self, data):
        x, y = data
        outputs = self.hf_model(x, labels=y, training=False)
        loss = outputs.loss
        return {"loss": loss}


class EvaluateCERCallback(tf.keras.callbacks.Callback):
    def __init__(self, valid_dataset, processor):
        super().__init__()
        self.valid_dataset = valid_dataset
        self.processor = processor

    def predict_batch(self, x, y):
        return self.model.hf_model(x, labels=y, training=False)

    def on_epoch_end(self, epoch, logs=None):
        total_cer = 0.0
        count = 0

        for x, y in self.valid_dataset:
            outputs = self.predict_batch(x, y)
            predicted_ids = tf.argmax(outputs.logits, axis=-1)

            pad_id = self.processor.tokenizer.pad_token_id
            y_fixed = tf.where(y < 0, pad_id, y)

            predicted_strs = self.processor.tokenizer.batch_decode(
                predicted_ids.numpy(), skip_special_tokens=True)
            ground_truth_strs = self.processor.tokenizer.batch_decode(
                y_fixed.numpy(), skip_special_tokens=True)

            for pred, ref in zip(predicted_strs, ground_truth_strs):
                total_cer += self.compute_cer(pred, ref)
                count += 1

        avg_cer = total_cer / count if count > 0 else 0.0
        print(f"Validation CER: {avg_cer:.4f}")
        if logs is not None:
            logs["val_cer"] = avg_cer

    def compute_cer(self, pred_str, ref_str):
        distance = editdistance.eval(pred_str, ref_str)
        return distance / len(ref_str) if len(ref_str) > 0 else 0.0

def main():
    preprocessed_path = "dataset/preprocessed"
    train_path = os.path.join(preprocessed_path, "train")
    valid_path = os.path.join(preprocessed_path, "valid")
    test_path = os.path.join(preprocessed_path, "test")

    # 1. 載入或下載資料
    if os.path.exists(train_path) and os.path.exists(valid_path) and os.path.exists(test_path):
        train_dataset = load_from_disk(train_path)
        valid_dataset = load_from_disk(valid_path)
        test_dataset = load_from_disk(test_path)
        print("載入已快取的資料集")
    else:
        (cv_zh_train, cv_zh_valid, cv_zh_test,
         cv_tai_train, cv_tai_valid, cv_tai_test) = load_commonvoice_datasets()

        train_dataset = merge_datasets(cv_zh_train, cv_tai_train, split_name="train")
        valid_dataset = merge_datasets(cv_zh_valid, cv_tai_valid, split_name="valid")
        test_dataset = merge_datasets(cv_zh_test, cv_tai_test, split_name="test")

        train_dataset, valid_dataset, test_dataset = preprocess_dataset(
            train_dataset, valid_dataset, test_dataset
        )

        os.makedirs(preprocessed_path, exist_ok=True)
        train_dataset.save_to_disk(train_path)
        valid_dataset.save_to_disk(valid_path)
        test_dataset.save_to_disk(test_path)
        print("已將資料集存到", preprocessed_path)

    # 2. 建立 vocab / tokenizer / processor
    vocab_json_path = "vocab.json"
    if not os.path.exists(vocab_json_path):
        print("建立 vocab.json ...")
        tokenizer, vocab_dict = create_and_save_vocab(train_dataset, vocab_json_path=vocab_json_path)
    else:
        print("使用現有 vocab.json ...")
        tokenizer = Wav2Vec2CTCTokenizer(
            vocab_json_path,
            unk_token="[UNK]",
            pad_token="[PAD]",
            word_delimiter_token="|"
        )
        with open(vocab_json_path, "r", encoding="utf-8") as f:
            vocab_dict = json.load(f)

    processor = get_processor(tokenizer)

    # 3. 檢查資料集中是否有超出範圍的字元
    # print("開始進行字元檢查...")
    # errors_train = debug_check_labels(train_dataset, processor)
    # errors_valid = debug_check_labels(valid_dataset, processor)
    # errors_test  = debug_check_labels(test_dataset, processor)

    # if errors_train or errors_valid or errors_test:
    #     print("發現超出範圍的字元，開始清理...")
    #     train_dataset = filter_invalid_chars(train_dataset, processor)
    #     valid_dataset = filter_invalid_chars(valid_dataset, processor)
    #     test_dataset  = filter_invalid_chars(test_dataset, processor)
    #     print("清理完成，重新檢查字元...")
    #     debug_check_labels(train_dataset, processor)
    # else:
    #     print("未發現超出範圍的字元，跳過清理步驟。")

    # 4. 將 dataset map 成 (input_values, labels)
    print("開始將 dataset map 成 (input_values, labels) ...")
    train_dataset = train_dataset.map(
        lambda batch: prepare_batch(batch, processor),
        remove_columns=train_dataset.column_names,
        batched=False
    )
    valid_dataset = valid_dataset.map(
        lambda batch: prepare_batch(batch, processor),
        remove_columns=valid_dataset.column_names,
        batched=False
    )
    test_dataset = test_dataset.map(
        lambda batch: prepare_batch(batch, processor),
        remove_columns=test_dataset.column_names,
        batched=False
    )

    # 5. 建立 tf.data.Dataset，並在生成器中檢查 label
    def gen_dataset(ds):
        for idx, item in enumerate(ds):
            x = np.array(item["input_values"], dtype=np.float32)
            y = np.array(item["labels"], dtype=np.int32)
            if y.max() >= processor.tokenizer.vocab_size:
                print(f"[gen_dataset] 第 {idx} 筆資料有超出範圍的標籤: {y.max()}, vocab_size={processor.tokenizer.vocab_size}")
                print("對應 labels:", y)
            yield (x, y)

    train_tf = tf.data.Dataset.from_generator(
        lambda: gen_dataset(train_dataset),
        output_types=(tf.float32, tf.int32),
        output_shapes=((None,), (None,))
    ).padded_batch(1, padded_shapes=([None], [None]), padding_values=(0.0, -1))

    valid_tf = tf.data.Dataset.from_generator(
        lambda: gen_dataset(valid_dataset),
        output_types=(tf.float32, tf.int32),
        output_shapes=((None,), (None,))
    ).padded_batch(1, padded_shapes=([None], [None]), padding_values=(0.0, -1))

    test_tf = tf.data.Dataset.from_generator(
        lambda: gen_dataset(test_dataset),
        output_types=(tf.float32, tf.int32),
        output_shapes=((None,), (None,))
    ).padded_batch(1, padded_shapes=([None], [None]), padding_values=(0.0, -1))

    # 6. 初始化預訓練模型並凍結指定層
    pretrained_model_name = "facebook/wav2vec2-base"
    tf.keras.mixed_precision.set_global_policy('float32')
    vocab_size = processor.tokenizer.vocab_size
    hf_model = TFWav2Vec2ForCTC.from_pretrained(
        pretrained_model_name,
        vocab_size=vocab_size,
        pad_token_id=processor.tokenizer.pad_token_id,
        from_pt=True
    )

    # 凍結特徵提取層
    hf_model.wav2vec2.feature_extractor.trainable = False

    # Transformer 層存取方式修改
    total_layers = len(hf_model.wav2vec2.encoder.layer)  # 使用 .layer
    layers_to_train = 3  # 保留最後 3 層
    freeze_until = total_layers - layers_to_train  # 凍結前 9 層

    for i, layer in enumerate(hf_model.wav2vec2.encoder.layer):
        if i < freeze_until:  # 前 9 層凍結
            layer.trainable = False
        else:  # 最後 3 層可訓練
            layer.trainable = True

    # 檢查凍結情況
    print("特徵提取層是否可訓練:", hf_model.wav2vec2.feature_extractor.trainable)
    for i, layer in enumerate(hf_model.wav2vec2.encoder.layer):
        print(f"Transformer 層 {i}: trainable = {layer.trainable}")
    print("lm_head 層是否可訓練:", hf_model.lm_head.trainable)

    # 7. 包成 Keras Model
    model = KerasWav2Vec2ForCTC(hf_model=hf_model, processor=processor)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(optimizer=optimizer)

    # 輸入音訊的取樣率是 16000
    model.build(input_shape=(None, 16000))
    # 模型摘要
    model.summary()

    # 8. 訓練迴圈
    num_epochs = 3
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        batch_num = 0
        total_loss = 0.0
        num_batches = 0

        for x, y in train_tf:
            try:
                loss_dict = model.train_step((x, y))
                total_loss += loss_dict['loss'].numpy()
                num_batches += 1
            except Exception as e:
                print(f"Error occurred at epoch {epoch+1}, batch {batch_num}")
                print("x shape:", x.shape)
                print("y shape:", y.shape)
                raise e
            if batch_num % 10 == 0:
                print(f"Epoch {epoch+1}, batch {batch_num}, loss: {loss_dict['loss'].numpy()}")
            batch_num += 1

        avg_train_loss = total_loss / num_batches if num_batches > 0 else 0.0
        print(f"Average training loss for epoch {epoch+1}: {avg_train_loss}")

        val_loss = model.evaluate(valid_tf, verbose=0)
        print(f"Validation loss after epoch {epoch+1}: {val_loss}")

    # 9. 測試集評估
    print("測試集評估:")
    test_loss = model.evaluate(test_tf)
    print(f"Test Loss: {test_loss}")

if __name__ == "__main__":
    main()