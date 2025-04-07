import os
from datasets import load_dataset, Audio, concatenate_datasets, load_from_disk
from tqdm.auto import tqdm
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor, TFWav2Vec2ForCTC
import json
import re
import tensorflow as tf
import numpy as np
import editdistance
import collections

# 啟用 mixed precision
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# 設定 GPU 記憶體動態配置
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

### 資料處理函數
def load_commonvoice_datasets():
    """
    載入 Common Voice 16.1 中文(臺灣) 與 臺語(閩南語) 的資料集，
    並分別返回訓練、驗證、測試集。
    """
    # 載入中文(臺灣)資料集
    cv_zh_train = load_dataset("mozilla-foundation/common_voice_16_1", "zh-TW", split="train", trust_remote_code=True)
    cv_zh_valid = load_dataset("mozilla-foundation/common_voice_16_1", "zh-TW", split="validation", trust_remote_code=True)
    cv_zh_test = load_dataset("mozilla-foundation/common_voice_16_1", "zh-TW", split="test", trust_remote_code=True)

    # 載入臺語(閩南語)資料集
    cv_tai_train = load_dataset("mozilla-foundation/common_voice_16_1", "nan-tw", split="train", trust_remote_code=True)
    cv_tai_valid = load_dataset("mozilla-foundation/common_voice_16_1", "nan-tw", split="validation", trust_remote_code=True)
    cv_tai_test = load_dataset("mozilla-foundation/common_voice_16_1", "nan-tw", split="test", trust_remote_code=True)
    
    return (cv_zh_train, cv_zh_valid, cv_zh_test, cv_tai_train, cv_tai_valid, cv_tai_test)

def merge_datasets(cv_zh, cv_tai, split_name="train"):
    """
    合併中文與臺語資料集，並回傳合併後的資料集
    """
    print(f"合併 {split_name} 資料集...")
    for _ in tqdm([1, 2], desc=f"合併 {split_name}"):
        pass
    merged_dataset = concatenate_datasets([cv_zh, cv_tai])
    return merged_dataset

def clean_sentence(example):
    """
    清理標籤中的全形括號及其內容
    """
    example['sentence'] = re.sub(r'（.*?）', '', example['sentence']).strip()
    return example

def preprocess_dataset(train_dataset, valid_dataset, test_dataset):
    """
    清理標籤，移除不必要欄位，只保留 audio 與 sentence，
    並將 audio 轉換成 16kHz 的取樣率
    """
    train_dataset = train_dataset.map(clean_sentence)
    valid_dataset = valid_dataset.map(clean_sentence)
    test_dataset = test_dataset.map(clean_sentence)
    
    keep_cols = ["audio", "sentence"]
    cols_to_remove = [col for col in train_dataset.column_names if col not in keep_cols]
    
    train_dataset = train_dataset.remove_columns(cols_to_remove)
    valid_dataset = valid_dataset.remove_columns(cols_to_remove)
    test_dataset = test_dataset.remove_columns(cols_to_remove)
    
    train_dataset = train_dataset.cast_column("audio", Audio(sampling_rate=16000))
    valid_dataset = valid_dataset.cast_column("audio", Audio(sampling_rate=16000))
    test_dataset = test_dataset.cast_column("audio", Audio(sampling_rate=16000))
    
    return train_dataset, valid_dataset, test_dataset

def build_vocab(train_dataset):
    """
    從訓練集中的 sentence 欄位建立字元集合
    """
    all_texts = " ".join(train_dataset["sentence"])
    vocab_chars = sorted(set(all_texts))
    return vocab_chars

def create_and_save_vocab(train_dataset, vocab_json_path="vocab.json"):
    """
    根據訓練資料建立字元集合、詞彙表字典，
    並存成 JSON 檔，同時建立 Wav2Vec2CTCTokenizer。
    """
    vocab_chars = build_vocab(train_dataset)
    print(f"字元總數: {len(vocab_chars)}")
    
    vocab_dict = {char: idx for idx, char in enumerate(vocab_chars)}
    
    if " " in vocab_dict:
        space_index = vocab_dict[" "]
        vocab_dict["|"] = space_index
        del vocab_dict[" "]
        print(f"將空格替換為 '|'，索引為 {space_index}")
    
    new_index = len(vocab_dict)
    vocab_dict["[UNK]"] = new_index
    vocab_dict["[PAD]"] = new_index + 1

    print(f"最終詞彙表大小: {len(vocab_dict)}")
    
    with open(vocab_json_path, "w", encoding="utf-8") as f:
        json.dump(vocab_dict, f, ensure_ascii=False)
    
    tokenizer = Wav2Vec2CTCTokenizer(vocab_json_path, unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
    return tokenizer, vocab_dict

def debug_check_labels(train_dataset, processor):
    """
    檢查訓練資料中所有句子的字元是否超出 vocab 範圍
    """
    vocab = processor.tokenizer.get_vocab()
    vocab_size = processor.tokenizer.vocab_size
    unk_id = processor.tokenizer.unk_token_id
    errors = []

    print("開始檢查訓練資料中所有句子的字元是否超出 vocab...")
    for i, sample in enumerate(tqdm(train_dataset, desc="檢查中")):
        sentence = sample["sentence"]
        labels = [vocab.get(char, unk_id) for char in sentence]
        for j, token in enumerate(labels):
            if token >= vocab_size:
                print(f"第 {i} 筆資料異常 -> 字元: '{sentence[j]}', token ID: {token}, vocab_size: {vocab_size}")
                errors.append((i, sentence, sentence[j], token))

    print(f"\n檢查完畢，共發現 {len(errors)} 筆異常。")
    return errors

def get_processor(tokenizer):
    """
    建立音訊特徵提取器並結合 tokenizer 成為處理器。
    """
    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, do_normalize=True)
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    return processor

def prepare_batch(batch, processor):
    """
    對單筆資料進行預處理：
    1. 音訊：提取 array 並送入處理器獲得 input_values
    2. 文字：逐字查表，若該字不存在則用 unk_token_id 取代
    """
    assert isinstance(batch["sentence"], str), "句子必須是字符串"
    
    audio = batch["audio"]
    inputs = processor(audio["array"], sampling_rate=audio["sampling_rate"], do_normalize=True)
    batch["input_values"] = inputs.input_values[0]
    batch["input_length"] = len(inputs.input_values[0])
    
    vocab = processor.tokenizer.get_vocab()
    vocab_size = processor.tokenizer.vocab_size
    unk_id = processor.tokenizer.unk_token_id
    
    labels = [vocab.get(char, unk_id) for char in batch["sentence"]]
    batch["labels"] = labels
    batch["labels_length"] = len(labels)
    
    assert all(0 <= token < vocab_size for token in labels), f"Label IDs out of range (vocab_size={vocab_size}): {labels}"
    return batch

class KerasWav2Vec2ForCTC(tf.keras.Model):
    def __init__(self, hf_model):
        super().__init__()
        self.hf_model = hf_model

    @tf.function(experimental_relax_shapes=True)
    def train_step(self, data):
        x, y = data

        # 檢查標籤值是否在合法範圍內
        max_label = tf.reduce_max(y)
        vocab_size = self.hf_model.config.vocab_size
        if max_label >= vocab_size:
            tf.print("發現無效標籤！")
            tf.print("最大標籤值:", max_label)
            tf.print("完整標籤內容:", y)
            raise ValueError(f"Label values must be <= vocab_size: {vocab_size}")

        with tf.GradientTape() as tape:
            outputs = self.hf_model(x, labels=y, training=True)
            loss = outputs.loss

            # 加入 get_scaled_loss
            scaled_loss = self.optimizer.get_scaled_loss(loss)

        # 先針對 scaled_loss 算梯度，再 unscale
        train_vars = self.trainable_variables
        scaled_grads = tape.gradient(scaled_loss, train_vars)
        grads = self.optimizer.get_unscaled_gradients(scaled_grads)

        self.optimizer.apply_gradients(zip(grads, train_vars))
        return {"loss": loss}

    @tf.function(experimental_relax_shapes=True)
    def test_step(self, data):
        """
        自訂測試步驟，跟 train_step 類似，但不需要做反向傳播。
        """
        x, y = data
        outputs = self.hf_model(x, labels=y, training=False)
        loss = outputs.loss
        return {"loss": loss}

class EvaluateCERCallback(tf.keras.callbacks.Callback):
    """
    這個 callback 會在每個 epoch 結束後，用驗證集計算 CER。
    """
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
            # 取得 logits -> decode
            predicted_ids = tf.argmax(outputs.logits, axis=-1)

            # decode 出預測字串
            predicted_strs = self.processor.tokenizer.batch_decode(
                predicted_ids.numpy(),
                skip_special_tokens=True
            )
            # 重新將 ground_truth 中的 -100 改為 pad_token_id
            mask = tf.equal(y, -100)
            indices = tf.where(mask)
            updates = tf.fill([tf.shape(indices)[0]],
                              self.processor.tokenizer.pad_token_id)
            ground_truth_ids = tf.tensor_scatter_nd_update(
                y, indices, updates
            )
            ground_truth_strs = self.processor.tokenizer.batch_decode(
                ground_truth_ids.numpy(),
                skip_special_tokens=True
            )
            # 計算 CER
            for pred, ref in zip(predicted_strs, ground_truth_strs):
                total_cer += self.compute_cer(pred, ref)
                count += 1

        avg_cer = total_cer / count if count > 0 else 0.0
        print(f"Validation CER: {avg_cer:.4f}")
        if logs is not None:
            logs["val_cer"] = avg_cer

    def compute_cer(self, pred_str, ref_str):
        # 這裡可以用你的自訂 CER 計算邏輯，或第三方函式庫
        import editdistance
        distance = editdistance.eval(pred_str, ref_str)
        return distance / len(ref_str) if len(ref_str) > 0 else 0.0

### 主程式
def main():
    preprocessed_path = "dataset/preprocessed"
    train_path = os.path.join(preprocessed_path, "train")
    valid_path = os.path.join(preprocessed_path, "valid")
    test_path = os.path.join(preprocessed_path, "test")
    
    if os.path.exists(train_path) and os.path.exists(valid_path) and os.path.exists(test_path):
        train_dataset = load_from_disk(train_path)
        valid_dataset = load_from_disk(valid_path)
        test_dataset = load_from_disk(test_path)
        print("載入已快取的資料集（包含前一次 VAD 結果）")
    else:
        (cv_zh_train, cv_zh_valid, cv_zh_test, cv_tai_train, cv_tai_valid, cv_tai_test) = load_commonvoice_datasets()
        train_dataset = merge_datasets(cv_zh_train, cv_tai_train, split_name="train")
        valid_dataset = merge_datasets(cv_zh_valid, cv_tai_valid, split_name="valid")
        test_dataset = merge_datasets(cv_zh_test, cv_tai_test, split_name="test")
        train_dataset, valid_dataset, test_dataset = preprocess_dataset(train_dataset, valid_dataset, test_dataset)
        
        os.makedirs(train_path, exist_ok=True)
        os.makedirs(valid_path, exist_ok=True)
        os.makedirs(test_path, exist_ok=True)
        
        train_dataset.save_to_disk(train_path)
        valid_dataset.save_to_disk(valid_path)
        test_dataset.save_to_disk(test_path)
        print("已將資料集存到", preprocessed_path)
    
    train_sample = train_dataset[0]
    print("第一筆訓練資料:")
    print("取樣率:", train_sample["audio"]["sampling_rate"])
    print("音訊長度:", len(train_sample["audio"]["array"]))
    print("句子:", train_sample["sentence"])

    tokenizer, vocab_dict = create_and_save_vocab(train_dataset)
    processor = get_processor(tokenizer)

    # 標記文件的路徑
    check_file = "labels_checked.txt"

    # 檢查標記文件
    if os.path.exists(check_file):
        with open(check_file, "r") as f:
            status = f.read().strip()
        if status == "passed":
            print("標籤檢查已通過，跳過檢查。")
        else:
            print("標籤檢查未通過，正在重新檢查...")
            errors = debug_check_labels(train_dataset, processor)
            if not errors:
                with open(check_file, "w") as f:
                    f.write("passed")
                print("標籤檢查通過，已更新標記文件。")
            else:
                with open(check_file, "w") as f:
                    f.write("failed")
                print("標籤檢查未通過，請修復問題。")
                return  # 終止程式
    else:
        print("未找到標記文件，正在執行標籤檢查...")
        errors = debug_check_labels(train_dataset, processor)
        if not errors:
            with open(check_file, "w") as f:
                f.write("passed")
            print("標籤檢查通過，已創建標記文件。")
        else:
            with open(check_file, "w") as f:
                f.write("failed")
            print("標籤檢查未通過，請修復問題。")
            return  # 終止程式

    train_dataset = train_dataset.map(lambda batch: prepare_batch(batch, processor), remove_columns=train_dataset.column_names)
    valid_dataset = valid_dataset.map(lambda batch: prepare_batch(batch, processor), remove_columns=valid_dataset.column_names)
    test_dataset = test_dataset.map(lambda batch: prepare_batch(batch, processor), remove_columns=test_dataset.column_names)

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
        padding_values=(0.0, -100)  # 修改為 -100
    )

    def valid_generator():
        for sample in valid_dataset:
            yield sample["input_values"], sample["labels"]

    valid_tfds = tf.data.Dataset.from_generator(valid_generator, output_signature=output_signature)
    valid_tfds = valid_tfds.padded_batch(
        batch_size=batch_size,
        padded_shapes=([None], [None]),
        padding_values=(0.0, -100)  # 修改為 -100
    )

    pretrained_model_name = "facebook/wav2vec2-base"
    tf.keras.mixed_precision.set_global_policy('float32')
    vocab_size = len(vocab_dict)  # 就是 max index + 1
    hf_model = TFWav2Vec2ForCTC.from_pretrained(
        pretrained_model_name,
        vocab_size=vocab_size,
        pad_token_id=tokenizer.pad_token_id,
        from_pt=True
    )

    print("Tokenizer vocab size:", processor.tokenizer.vocab_size)
    print("Model config vocab size:", hf_model.config.vocab_size)

    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    hf_model.wav2vec2.feature_extractor.trainable = False
    N = 3
    for layer in hf_model.wav2vec2.encoder.layer[:-N]:
        layer.trainable = False

    model = KerasWav2Vec2ForCTC(hf_model)
    trainable_params = np.sum([np.prod(var.shape) for var in model.trainable_variables])
    total_params = np.sum([np.prod(var.shape) for var in model.variables])
    print(f"可訓練參數/總參數: {trainable_params} / {total_params}")

    base_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    optimizer = tf.keras.mixed_precision.LossScaleOptimizer(base_optimizer)
    model.compile(optimizer=optimizer, run_eagerly=True)  # 啟用 Eager 模式進行調試

    # 檢查訓練資料集中的標籤
    print("檢查訓練資料集中的標籤...")
    for i, (x, y) in enumerate(train_tfds):
        max_label = tf.reduce_max(y).numpy()
        if max_label >= processor.tokenizer.vocab_size:
            print(f"第 {i} 個批次有無效標籤: {max_label}")
            print("標籤內容:", y.numpy())
            break
    else:
        print("所有批次的標籤都在合法範圍內。")

    cer_callback = EvaluateCERCallback(valid_tfds, processor)
    model.fit(
        train_tfds,
        epochs=3,
        callbacks=[cer_callback]
    )

if __name__ == "__main__":
    main()