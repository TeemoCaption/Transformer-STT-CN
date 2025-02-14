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
          config: 讀自 config.yaml 的設定字典
        """
        self.config = config
        # TSV (validated.tsv) 路徑
        self.tsv_path = config["data"]["tsv_path"]
        # 音檔資料夾
        self.audio_folder = config["data"]["audio_folder"]
        # 切分訓練與驗證集的比例
        self.test_size = config["data"]["test_size"]
        # 文字向量化時，最大序列長度
        self.max_target_len = config["data"]["max_target_len"]
        # joblib 多進程處理音檔數量
        self.n_jobs = config["parallel"]["n_jobs"]
        # 分批大小 (chunk_size)，一次處理多少音檔，避免 RAM 爆掉
        self.chunk_size = config["data"].get("chunk_size", 1000)

    def load_tsv_data(self):
        """
        從 TSV 檔案讀取所有的音檔路徑與對應中文句子。
        只讀取 "path", "sentence" 兩欄，組成一個 list[dict] 回傳。
        """
        df = pd.read_csv(self.tsv_path, sep="\t", usecols=["path","sentence"])
        data = []
        for _, row in df.iterrows():
            # 拼好完整音檔路徑
            audio_path = os.path.join(self.audio_folder, row["path"])
            data.append({"path": audio_path, "sentence": row["sentence"]})
        return data

    def path_to_spectrogram(self, full_path):
        """
        單一音檔 -> STFT 頻譜(1998, 257)。
        若失敗或音檔為空，回傳全零矩陣，以免後續流程爆炸。
        """
        max_duration = 10
        target_sr = 32000
        target_length = target_sr * max_duration
        frame_length = 400
        frame_step = 160
        fft_length = 512
        fft_bins = fft_length // 2 + 1
        expected_frames = 1 + (target_length - frame_length) // frame_step

        try:
            # librosa.load(): 讀取音檔成 NumPy 陣列
            # sr=32000 => 若原檔不是32000Hz，會做重採樣；mono=True => 轉單聲道
            audio, _ = librosa.load(full_path, sr=target_sr, mono=True)
        except:
            # 若讀檔出錯 => 回傳全零
            return np.zeros((expected_frames, fft_bins), dtype=np.float32)

        # 若讀到的陣列為空 => 也回傳全零
        if audio is None or len(audio) == 0:
            return np.zeros((expected_frames, fft_bins), dtype=np.float32)

        # 超過10秒就截斷, 不足則補零
        if len(audio) > target_length:
            audio = audio[:target_length]
        else:
            pad_len = target_length - len(audio)
            audio = np.pad(audio, (0,pad_len), mode="constant")

        # librosa.stft() => shape=(n_fft/2+1, frames)
        stft_np = librosa.stft(audio, n_fft=fft_length, hop_length=frame_step, win_length=frame_length)
        x = (np.abs(stft_np)**0.5).T  # => 轉置成 (frames, 257)

        # 對每個 frame 減均值 / 除標準差做標準化
        means = np.mean(x, axis=1, keepdims=True)
        stds = np.std(x, axis=1, keepdims=True)
        idx = (stds > 1e-6).reshape(-1)  # 避免 std=0
        x[idx, :] = (x[idx, :] - means[idx, :]) / (stds[idx, :] + 1e-6)

        # frames 若大於 1998 => 截斷；不足 => pad
        frames_now = x.shape[0]
        if frames_now > expected_frames:
            x = x[:expected_frames, :]
        elif frames_now < expected_frames:
            pad_frames = expected_frames - frames_now
            x = np.pad(x, ((0,pad_frames),(0,0)), mode="constant")

        return x.astype(np.float32)

    def preprocess_all_audio(self):
        """
        分段 (chunk) 方式前處理:
          1. 先把 TSV 所有檔案路徑讀進來
          2. 依 self.chunk_size 分塊
          3. 每塊用 joblib.Parallel 多進程並行算 STFT, 避免一次塞爆 RAM
          4. 把結果合併到 all_specs 後釋放該塊
          5. 最終回傳整個 data list (包含 spectrogram)
        """
        # 讀取 TSV => 得到 [ { "path":..., "sentence":... }, ... ]
        data = self.load_tsv_data()
        print(f"總共有 {len(data)} 筆音檔, chunk_size={self.chunk_size} 分批進行")

        all_specs = []  # 用來彙整所有檔案的結果
        start = 0
        # 依 chunk_size 做迴圈, 每次處理 chunk_size 個檔案
        while start < len(data):
            end = min(start + self.chunk_size, len(data))
            sub_data = data[start:end]

            # 並行處理 sub_data (小塊)
            print(f"處理 chunk: {start} ~ {end-1}, 共 {end-start} 個檔案")
            specs = Parallel(n_jobs=self.n_jobs)(
                delayed(self.path_to_spectrogram)(d["path"])
                for d in tqdm(sub_data, desc=f"Chunk {start}-{end-1}")
            )

            # 把算出來的 spectrogram 存回 sub_data
            for i, spec in enumerate(specs):
                sub_data[i]["spectrogram"] = spec

            # 合併到 all_specs
            all_specs.extend(sub_data)

            # 釋放 sub_data, specs, 讓垃圾回收避免佔記憶體
            del sub_data
            del specs

            # 移動到下一塊
            start = end

        return all_specs

    def split_data(self, data):
        """
        使用 sklearn.train_test_split 切出訓練集 / 驗證集
        """
        train_data, val_data = train_test_split(
            data, 
            test_size=self.test_size, 
            random_state=42
        )
        return train_data, val_data

    def build_vectorizer(self, train_data):
        """
        建立文字向量化器 VectorizeChar，用於把中文句子轉數字序列
        """
        sentences = [d["sentence"] for d in train_data]
        vectorizer = VectorizeChar(sentences, max_len=self.max_target_len)
        return vectorizer

    def to_tf_dataset(self, dataset_list, vectorizer, batch_size=4):
        """
        把已經在 memory 裡的 dataset_list（其中包含 "spectrogram" & "sentence"）轉成 tf.data.Dataset。
        
        1) 先把 spectrogram, sentence 全收進 NumPy array
        2) 用 tf.data.Dataset.from_tensor_slices(...) 將該陣列切片成多筆
        3) map => 把 (x, y) 包成 {"source": x, "target": y}
        4) batch => 指定每次多少筆, 便於後續訓練
        5) prefetch => 在 CPU/GPU 交換資料前先做預取, 提升效率
        """
        # 先把所有 spectrogram, text_seq 存到 list
        spectrograms = []
        text_seqs = []
        for d in dataset_list:
            spectrograms.append(d["spectrogram"])  # shape=(1998,257)
            text_seqs.append(vectorizer(d["sentence"]))

        # 轉成 NumPy array
        spectrograms = np.array(spectrograms, dtype=np.float32)
        text_seqs = np.array(text_seqs, dtype=np.int32)

        # 使用 tf.data.Dataset.from_tensor_slices( (spectrograms, text_seqs) )
        # => 會切成多筆資料, 每筆 (x, y)
        ds = tf.data.Dataset.from_tensor_slices((spectrograms, text_seqs))

        # ds.map(...): 把 (x, y) 轉成 {"source": x, "target": y} 這種 dict 格式
        # num_parallel_calls=tf.data.AUTOTUNE => 自動決定 map 操作使用的執行緒數量
        ds = ds.map(lambda x, y: {"source": x, "target": y}, num_parallel_calls=tf.data.AUTOTUNE)

        # ds.batch(...) => 每次取 batch_size 筆資料
        ds = ds.batch(batch_size)

        # ds.prefetch(tf.data.AUTOTUNE) => 在 CPU/GPU 間做 pipe 時可預先取下一批資料, 加快訓練吞吐量
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds
