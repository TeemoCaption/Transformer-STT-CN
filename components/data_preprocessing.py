import os  # 檔案與路徑操作模組
import pandas as pd  # 資料處理模組
import numpy as np  # 數值計算模組
import librosa  # 音訊處理模組
from joblib import Parallel, delayed  # 並行處理工具
from tqdm import tqdm  # 顯示進度條
from sklearn.model_selection import train_test_split  # 資料切分工具
import tensorflow as tf  # 深度學習框架
import gc  # 垃圾回收模組，用於強制釋放記憶體

from model.data_utils import VectorizeChar  # 文本向量化工具

class DataPreprocessing:
    def __init__(self, config):
        """
        初始化 DataPreprocessing 物件，並根據設定檔初始化各項參數

        參數：
        - config: dict，讀自 config.yaml 的設定字典，包含資料、訓練、平行處理等設定
        """
        self.config = config
        # TSV (validated.tsv) 檔案路徑
        self.tsv_path = config["data"]["tsv_path"]
        # 音檔資料夾路徑
        self.audio_folder = config["data"]["audio_folder"]
        # 切分訓練與驗證集的比例
        self.test_size = config["data"]["test_size"]
        # 文字向量化時，最大序列長度
        self.max_target_len = config["data"]["max_target_len"]
        # joblib 並行處理時開啟的 process 數量
        self.n_jobs = config["parallel"]["n_jobs"]
        # 分批處理的大小，一次處理多少音檔，避免 RAM 爆掉
        self.chunk_size = config["data"].get("chunk_size", 1000)
        # cache 資料夾路徑，若有設定則用來存放前處理後的 spectrogram cache
        self.cache_folder = config["data"].get("cache_folder", None)
        if self.cache_folder:
            os.makedirs(self.cache_folder, exist_ok=True)  # 若 cache 資料夾不存在則建立

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
            # 拼接完整的音檔路徑
            audio_path = os.path.join(self.audio_folder, row["path"])
            data.append({"path": audio_path, "sentence": row["sentence"]})
        return data

    def path_to_spectrogram(self, full_path):
        """
        將單一音檔轉換為 STFT 頻譜 (shape 為 (1998, 257))，
        若讀檔失敗或音檔為空則回傳全 0 矩陣，以避免後續流程錯誤

        參數：
        - full_path: str，音檔完整路徑

        回傳：
        - x.astype(np.float32): NumPy 陣列，經過處理後的 spectrogram，型態為 float32
        """
        max_duration = 10  # 固定處理 10 秒音檔
        target_sr = 32000  # 目標取樣率
        target_length = target_sr * max_duration  # 目標音檔長度
        frame_length = 400  # STFT 的視窗長度
        frame_step = 160    # STFT 的步長
        fft_length = 512    # FFT 的點數
        fft_bins = fft_length // 2 + 1  # STFT 後的頻率軸數量
        expected_frames = 1 + (target_length - frame_length) // frame_step  # 預期的時間軸幀數

        try:
            # 讀取音檔：使用 librosa.load 轉為 NumPy 陣列，若原始取樣率不同則重採樣至 target_sr，mono=True 轉單聲道
            audio, _ = librosa.load(full_path, sr=target_sr, mono=True)
        except:
            # 若讀檔出錯，回傳全 0 矩陣，避免後續錯誤
            return np.zeros((expected_frames, fft_bins), dtype=np.float32)

        if audio is None or len(audio) == 0:
            # 若讀到空音檔，回傳全 0 矩陣
            return np.zeros((expected_frames, fft_bins), dtype=np.float32)

        # 若音檔超過 target_length，則截斷；不足則補零
        if len(audio) > target_length:
            audio = audio[:target_length]
        else:
            pad_len = target_length - len(audio)
            audio = np.pad(audio, (0, pad_len), mode="constant")

        # 計算 STFT 頻譜，返回 shape 為 (n_fft/2+1, frames)
        stft_np = librosa.stft(audio, n_fft=fft_length, hop_length=frame_step, win_length=frame_length)
        # 將頻譜轉置為 (frames, 257) 並取幅值的 0.5 次方
        x = (np.abs(stft_np)**0.5).T

        # 對每個 frame 進行標準化：減均值除以標準差
        means = np.mean(x, axis=1, keepdims=True)
        stds = np.std(x, axis=1, keepdims=True)
        idx = (stds > 1e-6).reshape(-1)  # 避免標準差為 0 的情況
        x[idx, :] = (x[idx, :] - means[idx, :]) / (stds[idx, :] + 1e-6)

        # 若 frames 超過預期，則截斷；不足則補零
        frames_now = x.shape[0]
        if frames_now > expected_frames:
            x = x[:expected_frames, :]
        elif frames_now < expected_frames:
            pad_frames = expected_frames - frames_now
            x = np.pad(x, ((0, pad_frames), (0, 0)), mode="constant")

        return x.astype(np.float32)

    def preprocess_all_audio(self):
        """
        分段 (chunk) 方式前處理音檔：
          1. 讀取 TSV 檔案，取得所有音檔路徑與對應句子
          2. 若有設定 cache 且 cache 檔案存在，則直接載入 cache，不再前處理
          3. 若無 cache 則根據 chunk_size 分批處理音檔
          4. 每個 chunk 使用 joblib.Parallel 並行計算 STFT，並使用 threading backend 降低記憶體占用
          5. 每個 chunk 處理完後刪除暫存變數並呼叫 gc.collect() 強制釋放記憶體
          6. 前處理完成後，若有 cache_folder 設定，存檔至 cache 供下次直接載入
          7. 回傳包含 spectrogram 的整個 data list

        回傳：
        - all_specs: list，每筆資料為 {"path": 音檔路徑, "sentence": 中文句子, "spectrogram": STFT 頻譜}
        """
        # 讀取 TSV 檔案，取得所有資料
        data = self.load_tsv_data()
        print(f"總共有 {len(data)} 筆音檔, chunk_size={self.chunk_size} 分批進行")

        # 若有設定 cache_folder，檢查 cache 檔案是否存在
        cache_file = None
        if self.cache_folder:
            cache_file = os.path.join(self.cache_folder, "all_specs.npy")
            if os.path.exists(cache_file):
                print("發現 cache 檔案，直接載入 cache...")
                # np.load 參數說明：
                # - allow_pickle=True: 允許載入 Python 物件（例如 dict）
                all_specs = np.load(cache_file, allow_pickle=True)
                return all_specs.tolist()  # 轉換回 list

        all_specs = []  # 用來儲存所有音檔處理後的結果
        start = 0
        # 依據 chunk_size 逐批處理音檔
        while start < len(data):
            end = min(start + self.chunk_size, len(data))
            sub_data = data[start:end]

            # 並行處理該 chunk，使用 threading backend 以避免 pickle 大型物件造成記憶體問題
            print(f"處理 chunk: {start} ~ {end-1}, 共 {end-start} 個檔案")
            specs = Parallel(n_jobs=self.n_jobs, backend="threading")(
                delayed(self.path_to_spectrogram)(d["path"])
                for d in tqdm(sub_data, desc=f"Chunk {start}-{end-1}")
            )

            # 將處理後的 spectrogram 加入到各筆資料中
            for i, spec in enumerate(specs):
                sub_data[i]["spectrogram"] = spec

            # 合併該 chunk 結果至 all_specs
            all_specs.extend(sub_data)

            # 刪除不再使用的變數並強制呼叫垃圾回收，立即釋放記憶體
            del sub_data
            del specs
            gc.collect()

            # 進入下一個 chunk
            start = end

        # 若有設定 cache_folder，將結果存成 cache 以供下次直接載入
        if self.cache_folder:
            np.save(cache_file, np.array(all_specs, dtype=object))
            print(f"Cache 儲存至 {cache_file}")

        return all_specs

    def split_data(self, data):
        """
        使用 sklearn.train_test_split 將資料切分為訓練集與驗證集

        參數：
        - data: list，包含音檔資訊與 spectrogram 的資料列表

        回傳：
        - train_data, val_data: 切分後的訓練資料與驗證資料
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

        參數：
        - train_data: list，訓練資料列表，內含 "sentence" 欄位

        回傳：
        - vectorizer: VectorizeChar 物件
        """
        sentences = [d["sentence"] for d in train_data]
        vectorizer = VectorizeChar(sentences, max_len=self.max_target_len)
        return vectorizer

    def to_tf_dataset(self, dataset_list, vectorizer, batch_size=4):
        """
        將包含 "spectrogram" 與 "sentence" 的資料列表轉換為 tf.data.Dataset，
        以利後續訓練時批次讀取資料。

        參數：
        - dataset_list: list，每筆資料為 {"spectrogram": ..., "sentence": ...}
        - vectorizer: VectorizeChar 物件，用來將中文句子轉換成數字序列
        - batch_size: int，每個批次的資料筆數

        使用 TF 函式說明：
        - tf.data.Dataset.from_tensor_slices( (spectrograms, text_seqs) )
          參數：
            spectrograms: NumPy 陣列，型態為 float32，每筆音檔的頻譜資料
            text_seqs: NumPy 陣列，型態為 int32，每筆資料的數字化文字序列
          功能：根據輸入陣列切片成多筆資料，每筆為 (x, y)
        
        - ds.map(lambda x, y: {"source": x, "target": y}, num_parallel_calls=tf.data.AUTOTUNE)
          參數：
            lambda x, y: {"source": x, "target": y}: 將每筆 (x, y) 轉換為 dict 格式
            num_parallel_calls=tf.data.AUTOTUNE: 自動決定 map 操作使用的平行線程數量
          功能：將切片後的資料映射成訓練模型所需的格式
        
        - ds.batch(batch_size)
          參數：
            batch_size: 每個批次資料的筆數
          功能：將資料整理成批次，方便模型訓練
        
        - ds.prefetch(tf.data.AUTOTUNE)
          參數：
            tf.data.AUTOTUNE: 自動調整預取資料的數量
          功能：在 CPU 與 GPU 交換資料時，提前預取下一個批次，提高訓練效能

        回傳：
        - ds: tf.data.Dataset 物件，包含打包好的資料批次
        """
        spectrograms = []  # 儲存所有音檔的 spectrogram
        text_seqs = []     # 儲存所有文字數字序列
        for d in dataset_list:
            spectrograms.append(d["spectrogram"])  # 預期形狀 (1998, 257)
            text_seqs.append(vectorizer(d["sentence"]))

        # 將列表轉換為 NumPy 陣列
        spectrograms = np.array(spectrograms, dtype=np.float32)
        text_seqs = np.array(text_seqs, dtype=np.int32)

        # 將兩個 NumPy 陣列切片成多筆資料
        ds = tf.data.Dataset.from_tensor_slices((spectrograms, text_seqs))
        # 將每筆資料 (x, y) 映射成 dict 格式 {"source": x, "target": y}
        ds = ds.map(lambda x, y: {"source": x, "target": y}, num_parallel_calls=tf.data.AUTOTUNE)
        # 將資料整理成每批 batch_size 筆資料的批次
        ds = ds.batch(batch_size)
        # 預先加載下一批資料，提升資料交換效率
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds
