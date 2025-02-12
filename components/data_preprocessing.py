import pandas as pd
import os

class DataPreprocessing:
    def __init__(self, tsv_path, audio_folder):
        self.tsv_path = tsv_path
        self.audio_folder = audio_folder
        
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
            data.append({"audio": audio_path, "sentence": row["sentence"]})
        return data