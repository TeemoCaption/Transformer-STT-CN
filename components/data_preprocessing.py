import os
import pandas as pd
import numpy as np
import librosa
from joblib import Parallel, delayed
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import tensorflow as tf

from model.data_utils import VectorizeChar

class DataPreprocessing:
    def __init__(self, config):
        """
        參數:
          config: 從 config.yaml 讀入的整個設定字典
        """
        self.config = config
        self.tsv_path = config["data"]["tsv_path"]
        self.audio_folder = config["data"]["audio_folder"]
        self.test_size = config["data"]["test_size"]
        self.max_target_len = config["data"]["max_target_len"]
        self.n_jobs = config["parallel"]["n_jobs"]  # 多進程參數

        # 新增: cache_folder, 用來放離線儲存的 npy
        self.cache_folder = config["data"].get("cache_folder", "./spectrogram_cache")
        os.makedirs(self.cache_folder, exist_ok=True)

    def load_tsv_data(self):
        """
        讀取 tsv 檔案，取得所有 'path' 與對應 'sentence'
        回傳: 一個 list of dict, 如 [{"path": "...", "sentence": "..."}, ...]
        """
        df = pd.read_csv(self.tsv_path, sep="\t", usecols=["path","sentence"])
        data = []
        for _, row in df.iterrows():
            audio_path = os.path.join(self.audio_folder, row["path"])
            data.append({"path": audio_path, "sentence": row["sentence"]})
        return data

    def spectrogram_cache_path(self, full_path):
        """
        根據音檔路徑，生成一個對應的 .npy cache 檔名。
        例如 full_path=".../clips/common_voice_zh-TW_12345.mp3"
        就把 cache 取成 ".../spectrogram_cache/common_voice_zh-TW_12345.npy"
        """
        # 取檔名 (不含資料夾、附檔名)
        filename = os.path.splitext(os.path.basename(full_path))[0]
        return os.path.join(self.cache_folder, f"{filename}.npy")

    def path_to_spectrogram(self, full_path):
        """
        單檔音訊 -> STFT 頻譜 (shape = (1998, 257))
        若檔案讀取失敗，回傳 np.zeros((1998,257), dtype=np.float32)

        新增:
          - 如果 cache 檔存在, 直接讀取
          - 若不存在, 執行 stft, 再存檔
        """
        max_duration = 10
        target_sr = 32000
        target_length = target_sr * max_duration
        frame_length = 400
        frame_step = 160
        fft_length = 512
        fft_bins = fft_length // 2 + 1
        expected_frames = 1 + (target_length - frame_length) // frame_step

        # 檢查有沒有 .npy cache
        cache_file = self.spectrogram_cache_path(full_path)
        if os.path.exists(cache_file):
            # 已經算過 => 直接讀
            try:
                return np.load(cache_file)
            except:
                # 如果讀取 npy 出錯, 當成沒算過
                pass

        # 真正做 STFT
        try:
            audio, _ = librosa.load(full_path, sr=target_sr, mono=True)
        except:
            spec = np.zeros((expected_frames, fft_bins), dtype=np.float32)
            np.save(cache_file, spec)
            return spec

        if audio is None or len(audio) == 0:
            spec = np.zeros((expected_frames, fft_bins), dtype=np.float32)
            np.save(cache_file, spec)
            return spec

        if len(audio) > target_length:
            audio = audio[:target_length]
        else:
            pad_len = target_length - len(audio)
            audio = np.pad(audio, (0,pad_len), mode="constant")

        stft_np = librosa.stft(audio, n_fft=fft_length, hop_length=frame_step, win_length=frame_length)
        x = (np.abs(stft_np)**0.5).T  # (frames, 257)

        means = np.mean(x, axis=1, keepdims=True)
        stds = np.std(x, axis=1, keepdims=True)
        idx = (stds > 1e-6).reshape(-1)
        x[idx, :] = (x[idx, :] - means[idx, :]) / (stds[idx, :] + 1e-6)

        frames_now = x.shape[0]
        if frames_now > expected_frames:
            x = x[:expected_frames, :]
        elif frames_now < expected_frames:
            pad_frames = expected_frames - frames_now
            x = np.pad(x, ((0,pad_frames),(0,0)), mode="constant")

        spec = x.astype(np.float32)
        # 存到 cache
        np.save(cache_file, spec)
        return spec

    def preprocess_all_audio(self):
        """
        1) 讀取 tsv 取得所有檔案列表
        2) 利用 joblib 並行處理 -> 把每個檔案轉成 spectrogram numpy array (或直接讀 cache)
        3) data[i]["spectrogram"] = array
        4) 回傳整個 data list
        """
        data = self.load_tsv_data()
        print("開始並行處理所有音檔 (可讀/寫 cache)...")
        specs = Parallel(n_jobs=self.n_jobs)(
            delayed(self.path_to_spectrogram)(d["path"]) 
            for d in tqdm(data, desc="Audio Preprocessing")
        )
        for i, spec in enumerate(specs):
            data[i]["spectrogram"] = spec
        return data

    def split_data(self, data):
        """
        切分成 train / val dataset
        """
        train_data, val_data = train_test_split(
            data, 
            test_size=self.test_size, 
            random_state=42
        )
        return train_data, val_data

    def build_vectorizer(self, train_data):
        """
        建立文字向量化器
        """
        sentences = [d["sentence"] for d in train_data]
        vectorizer = VectorizeChar(sentences, max_len=self.max_target_len)
        return vectorizer

    def to_tf_dataset(self, dataset_list, vectorizer, batch_size=4):
        """
        把預先處理好的 dataset_list（含 "spectrogram" & "sentence"）轉成 tf.data.Dataset
        這裡直接從記憶體一次性做 Dataset。
        """
        spectrograms = []
        text_seqs = []
        for d in dataset_list:
            spectrograms.append(d["spectrogram"])  # np.array (1998,257)
            text_seq = vectorizer(d["sentence"])
            text_seqs.append(text_seq)

        spectrograms = np.array(spectrograms, dtype=np.float32)
        text_seqs = np.array(text_seqs, dtype=np.int32)

        ds = tf.data.Dataset.from_tensor_slices((spectrograms, text_seqs))
        ds = ds.map(lambda x, y: {"source": x, "target": y}, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(batch_size)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds
