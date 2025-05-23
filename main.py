import os
# 在 import TensorFlow 之前先關閉多餘的 log
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    TFWav2Vec2ForCTC
)
from tensorflow.keras import Model as KerasModel
from datasets import load_from_disk
import editdistance
import gc

from src.data_utils import (
    load_commonvoice_datasets,
    merge_datasets,
    preprocess_dataset,
    create_and_save_vocab,
)
from src.utils import get_processor, prepare_batch

# 讓 GPU 按需成長
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# 只保留 ERROR 訊息
tf.get_logger().setLevel('ERROR')

class KerasWav2Vec2ForCTC(KerasModel):
    """
    包裝 HF TFWav2Vec2ForCTC 的自訂 Keras Model，
    並在 train_step/test_step 做 label clamp。
    """
    def __init__(self, hf_model, processor):
        super().__init__()
        self.hf_model = hf_model
        self.processor = processor

    def call(self, inputs, training=False):
        return self.hf_model(inputs, training=training)

    def train_step(self, data):
        x, y = data
        pad_id = self.processor.tokenizer.pad_token_id
        vocab_size = self.hf_model.config.vocab_size

        # clamp：把 y < 0 或 y >= vocab_size 的都設成 pad_id
        y = tf.where(y < 0, pad_id, y)
        y = tf.where(y < vocab_size, y, pad_id)

        with tf.GradientTape() as tape:
            outputs = self.hf_model({"input_values": x, "labels": y}, training=True)
            loss = outputs.loss

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return {"loss": loss}

    def test_step(self, data):
        x, y = data
        pad_id = self.processor.tokenizer.pad_token_id
        vocab_size = self.hf_model.config.vocab_size

        y = tf.where(y < 0, pad_id, y)
        y = tf.where(y < vocab_size, y, pad_id)

        outputs = self.hf_model({"input_values": x, "labels": y}, training=False)
        loss = outputs.loss
        return {"loss": loss}

class EvaluateCERCallback(tf.keras.callbacks.Callback):
    """
    幫我們計算 Validation CER 的 Callback，
    之後手動呼叫 on_epoch_end。
    """
    def __init__(self, valid_dataset, processor):
        super().__init__()
        self.valid_dataset = valid_dataset
        self.processor = processor

    def on_epoch_end(self, epoch, logs=None):
        total_cer = 0.0
        count = 0
        pad_id = self.processor.tokenizer.pad_token_id
        vocab_size = self.model.hf_model.config.vocab_size

        for x, y in self.valid_dataset:
            y = tf.where(y < 0, pad_id, y)
            y = tf.where(y < vocab_size, y, pad_id)

            outputs = self.model.hf_model({"input_values": x, "labels": y}, training=False)
            pred_ids = tf.argmax(outputs.logits, axis=-1).numpy()

            preds = self.processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
            refs  = self.processor.tokenizer.batch_decode(y.numpy(), skip_special_tokens=True)

            for p, r in zip(preds, refs):
                dist = editdistance.eval(p, r)
                total_cer += (dist / len(r)) if len(r) > 0 else 0.0
                count += 1

        avg_cer = total_cer / count if count > 0 else 0.0
        print(f"Validation CER: {avg_cer:.4f}")
        if logs is not None:
            logs["val_cer"] = avg_cer


def main():
    # ─── 0. 目錄設置 ─────────────────────────────────
    model_root     = "model"
    pretrained_dir = os.path.join(model_root, "pretrained")
    os.makedirs(pretrained_dir, exist_ok=True)

    # ─── 1. 載入 or 處理資料集 ────────────────────────
    cache_dir = "dataset/preprocessed"
    train_p, valid_p, test_p = (
        os.path.join(cache_dir, name) for name in ("train","valid","test")
    )
    if os.path.exists(train_p) and os.path.exists(valid_p) and os.path.exists(test_p):
        train_ds = load_from_disk(train_p)
        valid_ds = load_from_disk(valid_p)
        test_ds  = load_from_disk(test_p)
        print("載入已快取資料集")
    else:
        zh_tr, zh_vl, zh_te, tai_tr, tai_vl, tai_te = load_commonvoice_datasets()
        train_ds = merge_datasets(zh_tr, tai_tr, split_name="train")
        valid_ds = merge_datasets(zh_vl, tai_vl, split_name="valid")
        test_ds  = merge_datasets(zh_te, tai_te, split_name="test")
        train_ds, valid_ds, test_ds = preprocess_dataset(train_ds, valid_ds, test_ds)
        os.makedirs(cache_dir, exist_ok=True)
        train_ds.save_to_disk(train_p)
        valid_ds.save_to_disk(valid_p)
        test_ds.save_to_disk(test_p)
        print("已儲存預處理後資料集")

    # ─── 2. Tokenizer & Processor ──────────────────────
    vocab_path = "vocab.json"
    if not os.path.exists(vocab_path):
        print("建立 vocab.json …")
        tokenizer, _ = create_and_save_vocab(train_ds, vocab_json_path=vocab_path)
    else:
        print("使用現有 vocab.json …")
        tokenizer = Wav2Vec2CTCTokenizer(
            vocab_path, unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|"
        )
    processor = get_processor(tokenizer)

    # ─── 3. 轉成 (input_values, labels) ─────────────────
    train_ds = train_ds.map(lambda b: prepare_batch(b, processor), remove_columns=train_ds.column_names, batched=False)
    valid_ds = valid_ds.map(lambda b: prepare_batch(b, processor), remove_columns=valid_ds.column_names, batched=False)
    test_ds  = test_ds.map(lambda b: prepare_batch(b, processor), remove_columns=test_ds.column_names, batched=False)

    # ─── 4. 建立 tf.data.Dataset ───────────────────────
    def gen(ds):
        for item in ds:
            yield (
                np.array(item["input_values"], dtype=np.float32),
                np.array(item["labels"],      dtype=np.int32)
            )

    def make_tf(ds):
        return tf.data.Dataset.from_generator(
            lambda: gen(ds),
            output_types=(tf.float32, tf.int32),
            output_shapes=((None,), (None,))
        ).padded_batch(1, padded_shapes=([None],[None]), padding_values=(0.0, -1))\
         .prefetch(tf.data.AUTOTUNE)

    train_tf = make_tf(train_ds)
    valid_tf = make_tf(valid_ds)
    test_tf  = make_tf(test_ds)

    # ─── 5. 轉換 or 載入 TF 模型 ────────────────────────
    if not os.listdir(pretrained_dir):
        print("首次將 PyTorch 權重轉成 TF…")
        hf_model = TFWav2Vec2ForCTC.from_pretrained(
            "facebook/wav2vec2-base",
            vocab_size=processor.tokenizer.vocab_size,
            pad_token_id=processor.tokenizer.pad_token_id,
            from_pt=True
        )
        hf_model.wav2vec2.feature_extractor.trainable = False
        total = len(hf_model.wav2vec2.encoder.layer)
        for i, layer in enumerate(hf_model.wav2vec2.encoder.layer):
            layer.trainable = (i >= total - 3)
        hf_model.save_pretrained(pretrained_dir)
        processor.save_pretrained(pretrained_dir)
        print("TF 模型已儲存")
    else:
        print("載入本地 TF 模型…")
        hf_model = TFWav2Vec2ForCTC.from_pretrained(pretrained_dir)
        processor = Wav2Vec2Processor.from_pretrained(pretrained_dir)
        hf_model.wav2vec2.feature_extractor.trainable = False
        total = len(hf_model.wav2vec2.encoder.layer)
        for i, layer in enumerate(hf_model.wav2vec2.encoder.layer):
            layer.trainable = (i >= total - 3)
        print("本地模型載入完成")

    # ─── 6. 包裝並編譯 Keras Model ─────────────────────
    model = KerasWav2Vec2ForCTC(hf_model=hf_model, processor=processor)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(optimizer=optimizer)
    model.build(input_shape=(None, 16000))
    model.summary()

    # ─── 7. 手動訓練迴圈，加入早停、記錄 Loss & CER ─────────────────
    epochs = 50
    patience = 3
    best_val_loss = float("inf")
    wait = 0

    train_losses = []
    val_losses   = []
    val_cers     = []
    cer_callback = EvaluateCERCallback(valid_tf, processor)

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        model.reset_metrics()

        # --- train ---
        total_loss = 0.0
        steps = 0
        for x, y in train_tf:
            out  = model.train_step((x, y))
            loss = out["loss"].numpy().item()
            total_loss += loss
            steps += 1
            if steps % 10 == 0:
                print(f"  [train] batch {steps}, loss: {loss:.4f}")
            del x, y, out, loss
            gc.collect()
        avg_train_loss = total_loss / steps
        print(f"  → Average training loss: {avg_train_loss:.4f}")

        # --- validation loss ---
        print("  → Start validation loss computation…")
        total_val = 0.0
        vsteps    = 0
        for x, y in valid_tf:
            out      = model.test_step((x, y))
            val_loss = out["loss"].numpy().item()
            total_val += val_loss
            vsteps   += 1
            if vsteps % 10 == 0:
                print(f"  [valid] batch {vsteps}, loss: {val_loss:.4f}")
            del x, y, out, val_loss
            gc.collect()
        avg_val_loss = total_val / vsteps
        print(f"  → Validation loss: {avg_val_loss:.4f}")
        print("  → Validation loss computation done")

        # 記錄 metrics
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        logs = {}
        cer_callback.model = model
        cer_callback.on_epoch_end(epoch, logs)
        val_cers.append(logs["val_cer"])

        # 早停判斷
        if avg_val_loss < best_val_loss:
            print("驗證損失有進步，儲存最佳權重！")
            best_val_loss = avg_val_loss
            wait = 0
            best_ckpt = os.path.join(model_root, "best_weights.h5")
            model.save_weights(best_ckpt)
        else:
            wait += 1
            print(f"{wait}/{patience} 個 epoch 沒進步")
            if wait >= patience:
                print(f"連續 {patience} 個 epoch 驗證損失都沒改善，觸發早停，停止訓練！")
                break

        # 存每 epoch 權重
        ckpt = os.path.join(model_root, f"weights_epoch_{epoch+1}.h5")
        model.save_weights(ckpt)
        print(f"  → 已儲存權重到 {ckpt}")

    # ─── 8. 繪製 Loss & CER 曲線 ─────────────────────────
    epochs_range = range(1, len(train_losses) + 1)
    
    plt.figure()
    plt.plot(epochs_range, train_losses, linestyle='-')
    plt.plot(epochs_range, val_losses,   linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train Loss', 'Validation Loss'])
    plt.title('Train vs Validation Loss per Epoch')
    plt.show()

    plt.figure()
    plt.plot(epochs_range, val_cers, linestyle='-')
    plt.xlabel('Epoch')
    plt.ylabel('Validation CER')
    plt.title('Validation CER per Epoch')
    plt.show()

    # ─── 9. 測試集評估 & HF 格式存檔 ───────────────────
    print("\n=== 最終測試集評估 ===")
    test_loss = model.evaluate(test_tf, verbose=0)
    print(f"Test Loss: {test_loss:.4f}")

    hf_model.save_pretrained(os.path.join(model_root, "finetuned_hf"))
    processor.save_pretrained(os.path.join(model_root, "finetuned_hf"))
    print("已儲存 Hugging Face 格式模型到 model/finetuned_hf")

if __name__ == "__main__":
    main()
