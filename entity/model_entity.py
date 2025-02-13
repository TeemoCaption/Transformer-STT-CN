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
        將音檔轉換為 STFT 頻譜圖 (librosa)，保證輸出固定形狀 (1998, 257)
        """
        max_duration = 10  # 固定 10 秒
        target_sr = 32000  # 強制使用 32000 Hz
        target_length = target_sr * max_duration  # = 320000
        frame_length = 400   # 相當於 win_length
        frame_step = 160     # 相當於 hop_length
        fft_length = 512     # n_fft
        fft_bins = fft_length // 2 + 1  # = 257
        expected_frames = 1 + (target_length - frame_length) // frame_step  # = 1998

        try:
            audio, sr = librosa.load(path, sr=target_sr, mono=True)
        except Exception as e:
            print(f"【錯誤】讀取 {path} 失敗：{e}")
            # 回傳全 0 頻譜，避免後續流程炸掉
            return tf.zeros([expected_frames, fft_bins], dtype=tf.float32)

        if audio is None or len(audio) == 0:
            print(f"【警告】檔案 {path} 音訊長度為 0，回傳空頻譜。")
            return tf.zeros([expected_frames, fft_bins], dtype=tf.float32)

        # Debug: 確認音檔資訊
        # print(f"[DEBUG] File: {path}")
        # print(f"        Original Audio Length = {len(audio)}, Sample Rate = {sr}")

        # 截斷或補零至 10 秒
        if len(audio) > target_length:
            audio = audio[:target_length]
        else:
            pad_len = target_length - len(audio)
            audio = np.pad(audio, (0, pad_len), mode='constant')

        # ===============================
        # 使用 librosa.stft 取頻譜
        # ===============================
        # shape: (n_fft/2+1, frames) = (257, frames)
        stft_np = librosa.stft(
            audio, n_fft=fft_length, hop_length=frame_step, win_length=frame_length
        )
        # 取幅度的 0.5 次方
        x = np.abs(stft_np)**0.5
        # 轉置後就會是 (frames, 257)
        x = x.T

        # 標準化：對每個 frame 做減均值、除標準差
        means = np.mean(x, axis=1, keepdims=True)
        stds = np.std(x, axis=1, keepdims=True)
        # 避免除以 0
        idx = (stds > 1e-6).reshape(-1)
        # 對非 0 的 frame 做標準化
        x[idx, :] = (x[idx, :] - means[idx, :]) / (stds[idx, :] + 1e-6)

        # 目前 x.shape = (frames, 257)
        frames_now = x.shape[0]

        # truncate 或補 zero
        if frames_now > expected_frames:
            x = x[:expected_frames, :]
        elif frames_now < expected_frames:
            pad_frames = expected_frames - frames_now
            x = np.pad(x, ((0, pad_frames), (0, 0)), mode='constant')

        # maybe shape = (1998, 257)
        #print(f"        Final Spectrogram Shape = {x.shape}")

        # 轉回 tf
        x_tf = tf.convert_to_tensor(x, dtype=tf.float32)
        # 靜態 shape，供後續 TF pipeline 驗證
        x_tf.set_shape([expected_frames, fft_bins])

        return x_tf

    def create_text_ds(self):
        texts = [d["sentence"] for d in self.data]
        text_seqs = [self.vectorizer(t) for t in texts]
        return tf.data.Dataset.from_tensor_slices(text_seqs)

    def create_audio_ds(self):
        audio_paths = []
        for d in self.data:
            if self.audio_folder in d["path"]:
                full_path = d["path"]
            else:
                full_path = os.path.join(self.audio_folder, d["path"])
            audio_paths.append(full_path)

        print("【除錯】檢查拼接後的音檔路徑：")
        for p in audio_paths:
            if os.path.exists(p):
                print(f"存在：{p}")
            else:
                print(f"不存在：{p}")

        # 濾掉不存在的檔案
        audio_paths = [p for p in audio_paths if os.path.exists(p)]
        audio_ds = tf.data.Dataset.from_tensor_slices(audio_paths)

        def load_audio(path):
            path = path.numpy().decode('utf-8')
            #print(f"【除錯】處理檔案：{path}")
            return self.path_to_audio(path)


        def load_audio_wrapper(path):
            audio = tf.py_function(func=load_audio, inp=[path], Tout=tf.float32)
            # 再次指定 shape，讓 TF dataset pipeline 知道輸出固定長度
            audio.set_shape([1998, 257])
            return audio

        audio_ds = audio_ds.map(load_audio_wrapper, num_parallel_calls=tf.data.AUTOTUNE)
        return audio_ds

    def create_tf_dataset(self, bs=4):
        audio_ds = self.create_audio_ds()
        text_ds = self.create_text_ds()
        ds = tf.data.Dataset.zip((audio_ds, text_ds))
        ds = ds.map(lambda x, y: {"source": x, "target": y})
        ds = ds.batch(bs)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds
