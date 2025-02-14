import os            # 檔案與路徑操作模組
import math          # 數學函式，例如 math.ceil()
import pandas as pd  # 資料處理模組
import numpy as np   # 數值計算模組
import librosa       # 音訊處理模組
from joblib import Parallel, delayed  # 並行處理工具（內部可用 threading）
from tqdm import tqdm           # 顯示進度條
from sklearn.model_selection import train_test_split  # 資料切分工具
import tensorflow as tf  # 深度學習框架
import gc  # 垃圾回收模組，用來強制釋放記憶體
import multiprocessing as mp  # 用於進程隔離

from model.data_utils import VectorizeChar  # 文本向量化工具

def process_chunk(chunk_idx, sub_data, config):
    """
    用於子進程中處理單個 chunk：
      1. 建立一個 DataPreprocessing 物件（各項參數從 config 載入）
      2. 對傳入的 sub_data (一個 list，每筆包含 "path" 與 "sentence")，
         依序呼叫 path_to_spectrogram() 轉換音檔，並把結果存回 dict 中
      3. 存成獨立的 npy 檔案：spectrogram_cache_folder/chunk_{chunk_idx}.npy
    參數：
      - chunk_idx: int，該 chunk 的索引
      - sub_data: list，該 chunk 中的音檔資料
      - config: dict，完整設定檔，子進程中用於建立 DataPreprocessing
    """
    # 在子進程中建立新的 DataPreprocessing 物件
    dp = DataPreprocessing(config)
    processed = []
    for d in tqdm(sub_data, desc=f"Chunk {chunk_idx} 處理中", leave=False):
        spec = dp.path_to_spectrogram(d["path"])
        d["spectrogram"] = spec
        processed.append(d)
    chunk_file = os.path.join(dp.spectrogram_cache_folder, f"chunk_{chunk_idx}.npy")
    np.save(chunk_file, np.array(processed, dtype=object))
    print(f"Chunk {chunk_idx} 已儲存於 {chunk_file}")

class DataPreprocessing:
    def __init__(self, config):
        """
        初始化 DataPreprocessing 物件，並根據設定檔初始化各項參數
        """
        self.config = config

        # 取得 audio_params 裡的音訊超參數
        audio_params = config["data"]["audio_params"]
        self.max_duration = audio_params["max_duration"]
        self.target_sr = audio_params["target_sr"]
        self.frame_length = audio_params["frame_length"]
        self.frame_step = audio_params["frame_step"]
        self.fft_length = audio_params["fft_length"]

        # 其餘參數與設定
        self.tsv_path = config["data"]["tsv_path"]
        self.audio_folder = config["data"]["audio_folder"]
        self.test_size = config["data"]["test_size"]
        self.max_target_len = config["data"]["max_target_len"]
        self.n_jobs = config["parallel"]["n_jobs"]
        self.chunk_size = config["data"].get("chunk_size", 1000)
        self.spectrogram_cache_folder = config["data"].get("spectrogram_cache_folder", None)
        if self.spectrogram_cache_folder:
            os.makedirs(self.spectrogram_cache_folder, exist_ok=True)

    def load_tsv_data(self):
        """
        從 TSV 檔案讀取所有音檔的路徑與對應中文句子，
        僅讀取 "path" 與 "sentence" 欄位，並回傳格式為 list[dict]
        回傳：
          - data: list，每筆為 {"path": 完整音檔路徑, "sentence": 中文句子}
        """
        df = pd.read_csv(self.tsv_path, sep="\t", usecols=["path", "sentence"])
        data = []
        for _, row in df.iterrows():
            audio_path = os.path.join(self.audio_folder, row["path"])
            data.append({"path": audio_path, "sentence": row["sentence"]})
        return data

    def path_to_spectrogram(self, full_path):
        """
        將單一音檔轉換為 STFT 頻譜，若讀檔失敗或音檔為空則回傳全 0 矩陣
        """
        max_duration = self.max_duration
        target_sr = self.target_sr
        frame_length = self.frame_length
        frame_step = self.frame_step
        fft_length = self.fft_length

        target_length = target_sr * max_duration
        fft_bins = fft_length // 2 + 1
        expected_frames = 1 + (target_length - frame_length) // frame_step

        try:
            audio, _ = librosa.load(full_path, sr=target_sr, mono=True)
        except:
            return np.zeros((expected_frames, fft_bins), dtype=np.float32)
        if audio is None or len(audio) == 0:
            return np.zeros((expected_frames, fft_bins), dtype=np.float32)
        if len(audio) > target_length:
            audio = audio[:target_length]
        else:
            pad_len = target_length - len(audio)
            audio = np.pad(audio, (0, pad_len), mode="constant")
        stft_np = librosa.stft(audio, n_fft=fft_length, hop_length=frame_step, win_length=frame_length)
        x = (np.abs(stft_np)**0.5).T
        means = np.mean(x, axis=1, keepdims=True)
        stds = np.std(x, axis=1, keepdims=True)
        idx = (stds > 1e-6).reshape(-1)
        x[idx, :] = (x[idx, :] - means[idx, :]) / (stds[idx, :] + 1e-6)
        frames_now = x.shape[0]
        if frames_now > expected_frames:
            x = x[:expected_frames, :]
        elif frames_now < expected_frames:
            pad_frames = expected_frames - frames_now
            x = np.pad(x, ((0, pad_frames), (0, 0)), mode="constant")
        return x.astype(np.float32)

    def chunk_preprocess_and_save(self):
        """
        以 chunk_size 為單位前處理音檔：
          1. 讀取所有音檔資料，依據 chunk_size 分成多個 chunk
          2. 對於每個 chunk，先檢查是否已存在對應的 npy 檔案
             - 若存在，則跳過處理該 chunk
             - 若不存在，啟動一個獨立子進程進行前處理與存檔
          3. 子進程結束後，該 chunk 占用的記憶體會由 OS 回收

        這樣下次訓練時會接續處理缺少的 chunk，而不必重複前處理已存在的資料。
        """
        data = self.load_tsv_data()
        total_chunks = math.ceil(len(data) / self.chunk_size)
        print(f"總共有 {len(data)} 筆音檔，將以 chunk_size={self.chunk_size} 分批處理，共 {total_chunks} 個 chunk。")

        for chunk_idx in tqdm(range(total_chunks), desc="整體處理進度"):
            # 檢查此 chunk 檔案是否已存在
            chunk_file = os.path.join(self.spectrogram_cache_folder, f"chunk_{chunk_idx}.npy")
            if os.path.exists(chunk_file):
                print(f"Chunk {chunk_idx} 已存在，跳過處理。")
                continue

            start = chunk_idx * self.chunk_size
            end = min(start + self.chunk_size, len(data))
            sub_data = data[start:end]

            # 為每個 chunk 啟動一個獨立子進程進行處理
            p = mp.Process(target=process_chunk, args=(chunk_idx, sub_data, self.config))
            p.start()
            p.join()  # 等待子進程結束
            gc.collect()

        print("所有缺少的 chunk 處理完畢！")

    def preprocess_all_audio(self):
        """
        如果希望一次性載入所有前處理結果（例如後續 train/val 切分），
        本方法會先檢查是否存在整合後的 cache 檔案（all_specs.npy），
        若不存在則合併所有獨立 chunk 的資料，最後回傳整合後的清單。
        回傳：
          - all_specs: list，每筆資料為 {"path": 音檔路徑, "sentence": 中文句子, "spectrogram": STFT 頻譜}
        """
        cache_file = None
        if self.spectrogram_cache_folder:
            cache_file = os.path.join(self.spectrogram_cache_folder, "all_specs.npy")
            if os.path.exists(cache_file):
                print("發現整合 cache 檔案，直接載入...")
                all_specs = np.load(cache_file, allow_pickle=True)
                return all_specs.tolist()
        all_specs = []
        for file in sorted(os.listdir(self.spectrogram_cache_folder)):
            if file.startswith("chunk_") and file.endswith(".npy"):
                chunk_data = np.load(os.path.join(self.spectrogram_cache_folder, file), allow_pickle=True)
                all_specs.extend(chunk_data.tolist())
        if self.spectrogram_cache_folder:
            np.save(cache_file, np.array(all_specs, dtype=object))
            print(f"整合 cache 儲存於 {cache_file}")
        return all_specs

    def split_data(self, data):
        """
        使用 sklearn.train_test_split 將資料切分為訓練集與驗證集
        """
        train_data, val_data = train_test_split(
            data, 
            test_size=self.test_size, 
            random_state=42
        )
        return train_data, val_data

    def build_vectorizer(self, train_data):
        """
        建立文字向量化器 VectorizeChar，用於將中文句子轉換為數字序列
        """
        sentences = [d["sentence"] for d in train_data]
        vectorizer = VectorizeChar(sentences, max_len=self.max_target_len)
        return vectorizer

    def to_tf_dataset(self, dataset_list, vectorizer, batch_size=4):
        """
        將包含 "spectrogram" 與 "sentence" 的資料列表轉換為 tf.data.Dataset，
        以利後續訓練時批次讀取資料
        """
        spectrograms = []
        text_seqs = []
        for d in dataset_list:
            spectrograms.append(d["spectrogram"])
            text_seqs.append(vectorizer(d["sentence"]))
        spectrograms = np.array(spectrograms, dtype=np.float32)
        text_seqs = np.array(text_seqs, dtype=np.int32)
        ds = tf.data.Dataset.from_tensor_slices((spectrograms, text_seqs))
        ds = ds.map(lambda x, y: {"source": x, "target": y}, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(batch_size)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds
