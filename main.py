# main.py

import os
import subprocess
import tensorflow as tf

# 從 utils 模組引入數據前處理相關函數
from src.utils import process_tsv
# 從預訓練模型模組引入模型與訓練函數
from models.pretraining import Wav2Vec2Model, contrastive_loss

import glob

# -------------------------------
# 數據前處理部分
# -------------------------------

# 設定資料夾路徑
dataset_dir = 'dataset'
clips_dir = os.path.join(dataset_dir, 'clips')
output_dir = os.path.join(dataset_dir, 'processed')

# TSV 檔案列表
tsv_files = [
    os.path.join(dataset_dir, 'train.tsv'),
    os.path.join(dataset_dir, 'validated.tsv'),
    os.path.join(dataset_dir, 'test.tsv')
]

def run_preprocessing():
    """
    呼叫數據前處理函數，將原始 MP3 轉換成統一格式的 WAV 檔，存放在 output_dir 中
    """
    for tsv_file in tsv_files:
        print(f"開始處理 {os.path.basename(tsv_file)} ...")
        process_tsv(tsv_file, clips_dir, output_dir)
        print(f"{os.path.basename(tsv_file)} 處理完成。\n")

# -------------------------------
# 預訓練部分：建立資料集與訓練流程
# -------------------------------

def load_wav(filename):
    """
    利用 tf.io 讀取 WAV 檔（假設格式為 16kHz 單聲道）
    """
    audio_binary = tf.io.read_file(filename)
    waveform, sample_rate = tf.audio.decode_wav(audio_binary, desired_channels=1)
    # 壓縮掉 channel 維度，使 waveform 形狀變成 (samples,)
    return tf.squeeze(waveform, axis=-1)

def get_dataset(processed_dir, batch_size=1):
    """
    從 processed_dir 讀取所有 WAV 檔，建立 tf.data.Dataset 並進行批次化與隨機打亂
    """
    wav_files = glob.glob(os.path.join(processed_dir, "*.wav"))
    ds = tf.data.Dataset.from_tensor_slices(wav_files)
    ds = ds.map(lambda x: load_wav(x), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.shuffle(1000)
    # 由於每個音檔長度可能不同，使用 padded_batch 補齊
    ds = ds.padded_batch(batch_size, padded_shapes=[None])
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

def train_pretraining(dataset, epochs=10, batch_size=8):
    """
    建立模型並執行預訓練
    """
    # 建立模型（根據需求可調整模型參數）
    model = Wav2Vec2Model()
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        for step, batch in enumerate(dataset):
            # 如果輸入 shape 為 (batch, time)，補上 channel 維度
            if tf.rank(batch) == 2:
                batch = tf.expand_dims(batch, -1)
            with tf.GradientTape() as tape:
                # 訓練模式下模型會進行連續遮罩處理
                context, quantized = model(batch, training=True)
                # 此處需提供遮罩資訊來計算對比損失
                # 範例中以隨機生成的 dummy_mask 來模擬（實際上你可從 ContiguousMasking 層中取得）
                batch_size_actual = tf.shape(context)[0]
                seq_len = tf.shape(context)[1]
                dummy_mask = tf.cast(tf.random.uniform([batch_size_actual, seq_len]) < 0.065, tf.bool)
                loss = contrastive_loss(context, quantized, dummy_mask)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            if step % 10 == 0:
                print(f"Step {step}: Loss = {loss.numpy():.4f}")
        # 每個 epoch 結束後可儲存權重
        model.save_weights(f"checkpoints/epoch_{epoch+1}.ckpt")
    # 儲存最終模型
    model.save("final_wav2vec2_model")
    print("預訓練完成！")

def run_pretraining():
    """
    從預處理後的資料夾建立資料集並執行預訓練流程
    """
    batch_size = 8
    epochs = 20
    dataset = get_dataset(output_dir, batch_size)
    train_pretraining(dataset, epochs=epochs, batch_size=batch_size)

# -------------------------------
# 主流程
# -------------------------------
if __name__ == '__main__':
    # print("=== 開始數據前處理 ===")
    # run_preprocessing()
    print("=== 數據前處理完成，開始預訓練 ===")
    run_pretraining()
