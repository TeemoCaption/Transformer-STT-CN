import pandas as pd
import os
from entity.model_entity import CreateTensors
from model.data_utils import VectorizeChar
from sklearn.model_selection import train_test_split

class DataPreprocessing:
    def __init__(self, tsv_path, audio_folder, test_size=0.2, random_state=42):
        self.tsv_path = tsv_path
        self.audio_folder = audio_folder
        self.test_size = test_size  
        self.random_state = random_state
        
    def load_tsv_data(self):
        """
        讀取 tsv 檔案，取得音檔路徑與對應句子
        參數：
        - tsv_path: TSV 檔案路徑
        - audio_folder: 音檔存放的資料夾
        回傳：
        - data: 包含 {'path': '完整音檔路徑', 'sentence': '中文句子'} 的列表
        """
        df = pd.read_csv(self.tsv_path, sep="\t", usecols=['path', 'sentence'])  # 只讀取 path 和 sentence
        data = []
        for _, row in df.iterrows():
            audio_path = os.path.join(self.audio_folder, row["path"])  # 取得完整音檔路徑
            data.append({"path": audio_path, "sentence": row["sentence"]})
        return data
    
    def split_data(self):
        """
        將數據集劃分為訓練集和驗證集
        回傳：
        - train_data: 訓練集
        - val_data: 驗證集
        """
        data = self.load_tsv_data()
        train_data, val_data = train_test_split(data, test_size=self.test_size, random_state=self.random_state)
        return train_data, val_data
