import os
import librosa
import numpy as np
import tensorflow as tf
from model.data_utils import VectorizeChar

class CreateTensors:
    def __init__(self, data: list, vectorizer: VectorizeChar, audio_folder: str):
        self.data = data
        self.vectorizer = vectorizer
        self.audio_folder = audio_folder
    
    def path_to_audio(self, path):
        """
        將音檔轉換為 stft 頻譜圖\n
        參數：\n
        path: 音檔路徑
        """
        # 使用 librosa 讀取 MP3 檔案，sr=None 保持原取樣率
        audio, sample_rate = librosa.load(path, sr=None, mono=True)

        # 設定最大長度（10 秒）
        max_duration = 10
        target_length = int(sample_rate * max_duration)  # 計算最大取樣數

        # 確保音訊長度符合標準
        if len(audio) > target_length:
            audio = audio[:target_length]  # 截斷
        else:
            pad_length = target_length - len(audio)
            audio = np.pad(audio, (0, pad_length), mode='constant')  # 填充

        # 根據取樣率調整 STFT 參數
        if sample_rate == 16000:  # 語音數據（16kHz）
            frame_length = 400
            frame_step = 160
            fft_length = 512
            pad_len = 2754
        elif sample_rate == 44100:  # 音樂數據（44.1kHz）
            frame_length = 1024
            frame_step = 512
            fft_length = 1024
            pad_len = 4000
        else:  # 預設 16kHz
            frame_length = 400
            frame_step = 160
            fft_length = 512
            pad_len = 2754

        # 轉換為 TensorFlow 張量
        audio = tf.convert_to_tensor(audio, dtype=tf.float32)

        # 計算 STFT
        stfts = tf.signal.stft(audio, frame_length=frame_length, frame_step=frame_step, fft_length=fft_length)

        # 取 STFT 絕對值開平方
        x = tf.math.pow(tf.abs(stfts), 0.5)

        # 標準化
        means = tf.math.reduce_mean(x, 1, keepdims=True)
        stddevs = tf.math.reduce_std(x, 1, keepdims=True)
        x = (x - means) / stddevs
        
        # 填充至 10 秒
        paddings = tf.constant([[0, pad_len], [0, 0]])
    
        x = tf.pad(x, paddings, "CONSTANT")[:pad_len, :]

        return x
    
    def create_text_ds(self):
        """
        創建文本數據集
        """
        texts = [d["sentence"] for d in self.data]  # 取得每條數據的文本
        # 將每個文本轉換為數字序列
        text_ds = [self.vectorizer(t) for t in texts]
        # tf.data.Dataset.from_tensor_slices 創建數據集
        text_ds = tf.data.Dataset.from_tensor_slices(text_ds)
        return text_ds
    
    def create_audio_ds(self):
        """
        創建音頻數據集
        """
        # 音檔完整路徑
        audio_paths = [os.path.join(self.audio_folder, d["path"]) for d in self.data]  

        # 檢查音檔是否存在
        audio_paths = [p if os.path.exists(p) else None for p in audio_paths]

        # 過濾掉 None 值（找不到音檔的情況）
        audio_paths = list(filter(None, audio_paths))

        # 轉換為 TensorFlow Dataset
        audio_ds = tf.data.Dataset.from_tensor_slices(audio_paths)
        audio_ds = audio_ds.map(
            self.path_to_audio, num_parallel_calls=tf.data.AUTOTUNE  # 使用多線程加速
        )
        return audio_ds

    def create_tf_dataset(self, bs=4):
        """
        創建數據集\n
        參數：\n
        - data: 數據\n
        - vectorizer: 用於文本向量化的實例\n
        - bs: 批次大小
        """
        audio_ds = self.create_audio_ds()
        text_ds = self.create_text_ds()
        # tf.data.Dataset.zip 將兩個數據集合併
        ds = tf.data.Dataset.zip((audio_ds, text_ds))
        # source: 音頻，target: 文本
        ds = ds.map(lambda x, y: {"source": x, "target": y})
        # batch 是將數據集分成多個批次，每個批次的大小是 bs
        ds = ds.batch(bs)
        # prefetch 是用來加速數據集的方法，tf.data.AUTOTUNE 表示自動選擇最佳的參數
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds
