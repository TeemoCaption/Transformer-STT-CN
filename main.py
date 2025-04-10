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

# 啟用混合精度
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# 設定 GPU 記憶體動態配置
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

def fix_weight_norm_layers(model):
    """
    遍歷模型所有層，若層類型名稱中包含 "WeightNormConv1D"，則強制將其 weight_v 與 weight_g
    轉換成 float32。這是為了避免在混合精度運算中，權重型別不一致的問題。
    """
    for layer in model.layers:
        if "WeightNormConv1D" in layer.__class__.__name__:
            if hasattr(layer, "weight_v"):
                # 將 weight_v 強制轉 float32
                layer.weight_v.assign(tf.cast(layer.weight_v, tf.float32))
            if hasattr(layer, "weight_g"):
                # 將 weight_g 強制轉 float32
                layer.weight_g.assign(tf.cast(layer.weight_g, tf.float32))
        # 若該層內部還有子層，遞迴處理
        if hasattr(layer, "layers") and layer.layers:
            fix_weight_norm_layers(layer)

class KerasWav2Vec2ForCTC(tf.keras.Model):
    def __init__(self, hf_model, processor):
        super().__init__()
        self.hf_model = hf_model
        self.processor = processor

    @tf.function(experimental_relax_shapes=True)
    def train_step(self, data):
        x, y = data

        # 將 -1 替換為 0，避免 CTC loss 出錯
        y = tf.where(y < 0, 0, y)

        # 檢查標籤值是否在合法範圍內
        max_label = tf.reduce_max(y)
        tf.print("Train step: max_label =", max_label)
        tf.print("Labels shape =", tf.shape(y))
        tf.print("Labels sample =", y[:10])

        if max_label >= self.hf_model.config.vocab_size:
            tf.print("【DEBUG】發現無效標籤 (超出 vocab_size):", max_label)
            raise ValueError(f"標籤值必須 <= vocab_size: {self.hf_model.config.vocab_size}")

        with tf.GradientTape() as tape:
            outputs = self.hf_model(x, labels=y, training=True)
            loss = outputs.loss
            scaled_loss = self.optimizer.get_scaled_loss(loss)

        train_vars = self.trainable_variables
        scaled_grads = tape.gradient(scaled_loss, train_vars)
        grads = self.optimizer.get_unscaled_gradients(scaled_grads)
        self.optimizer.apply_gradients(zip(grads, train_vars))

        return {"loss": loss}

    @tf.function(experimental_relax_shapes=True)
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

    @tf.function(experimental_relax_shapes=True)
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
    print("開始進行字元檢查...")
    errors_train = debug_check_labels(train_dataset, processor)
    errors_valid = debug_check_labels(valid_dataset, processor)
    errors_test  = debug_check_labels(test_dataset, processor)

    # 只有當有錯誤存在時才執行清理和再次檢查
    if errors_train or errors_valid or errors_test:
        print("發現超出範圍的字元，開始清理...")
        train_dataset = filter_invalid_chars(train_dataset, processor)
        valid_dataset = filter_invalid_chars(valid_dataset, processor)
        test_dataset  = filter_invalid_chars(test_dataset, processor)
        print("清理完成，重新檢查字元...")
        debug_check_labels(train_dataset, processor)
    else:
        print("未發現超出範圍的字元，跳過清理步驟。")

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
    ).padded_batch(4, padded_shapes=([None], [None]), padding_values=(0.0, -1))

    valid_tf = tf.data.Dataset.from_generator(
        lambda: gen_dataset(valid_dataset),
        output_types=(tf.float32, tf.int32),
        output_shapes=((None,), (None,))
    ).padded_batch(2, padded_shapes=([None], [None]), padding_values=(0.0, -1))

    test_tf = tf.data.Dataset.from_generator(
        lambda: gen_dataset(test_dataset),
        output_types=(tf.float32, tf.int32),
        output_shapes=((None,), (None,))
    ).padded_batch(2, padded_shapes=([None], [None]), padding_values=(0.0, -1))

    # 6. 初始化 HF 模型 (注意 vocab_size 必須與 tokenizer 相符)
    config_kwargs = {
        "vocab_size": processor.tokenizer.vocab_size,
        "activation_dropout": 0.1,
        "attention_dropout": 0.1,
        "hidden_dropout": 0.1,
    }
    from transformers import Wav2Vec2Config
    wav2vec_config = Wav2Vec2Config(**config_kwargs)

    hf_model = TFWav2Vec2ForCTC(config=wav2vec_config)
    # 新增：修正 HF 模型中 weight normalization conv 層的權重 dtype
    fix_weight_norm_layers(hf_model)

    # 7. 包成 Keras Model
    model = KerasWav2Vec2ForCTC(hf_model=hf_model, processor=processor)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
    model.compile(optimizer=optimizer)

    # 為了方便調試，暫時讓 tf.function 以 eager mode 運行，
    # 這樣可以讓 tf.print 印出詳細訊息，調試完畢後可關閉
    tf.config.run_functions_eagerly(True)

    # 8. 加入自訂 Callback 計算 CER
    cer_callback = EvaluateCERCallback(valid_tf, processor)

    # 9. 開始訓練
    model.fit(
        train_tf,
        validation_data=valid_tf,
        epochs=3,
        callbacks=[cer_callback]
    )

    # 10. 測試
    print("測試集評估:")
    test_loss = model.evaluate(test_tf)
    print(f"Test Loss: {test_loss}")

if __name__ == "__main__":
    main()
