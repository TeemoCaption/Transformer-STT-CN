import os
import pandas as pd
import json
import re
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import h5py

class DataUtils:
    def __init__(self, dataset_folder, limit=None):
        """初始化資料處理類別"""
        self.dataset_folder = dataset_folder
        self.limit = limit

    def load_data(self):
        """載入並分割資料集"""
        train_df = pd.read_csv(os.path.join(self.dataset_folder, "train.tsv"), sep='\t')
        valid_df = pd.read_csv(os.path.join(self.dataset_folder, "validated.tsv"), sep='\t')
        test_df = pd.read_csv(os.path.join(self.dataset_folder, "test.tsv"), sep='\t')
        combined_df = pd.concat([train_df, valid_df], ignore_index=True)
        if self.limit:
            combined_df = combined_df.head(self.limit)
        cols_to_keep = ['path', 'sentence']
        combined_df = combined_df[cols_to_keep]
        test_df = test_df[cols_to_keep]
        new_train_df, new_valid_df = train_test_split(combined_df, test_size=0.2, random_state=42)
        print(f"合併後總筆數: {len(combined_df)} 筆\n新訓練集筆數: {len(new_train_df)} 筆\n新驗證集筆數: {len(new_valid_df)} 筆\nTest 資料筆數: {len(test_df)} 筆")
        return new_train_df, new_valid_df, test_df

    def clean_sentence(self, sentence):
        """清理句子，只保留中文字符"""
        return re.sub(r"[^\u4e00-\u9fff]", "", sentence)

    def create_vocab(self, sentences, output_path):
        """建立詞彙表並儲存"""
        vocab_set = {"<PAD>", "<SOS>", "<EOS>", "<MASK>"}
        for sentence in sentences:
            vocab_set.update(self.clean_sentence(sentence))
        vocab = sorted(vocab_set)
        word2idx = {word: idx for idx, word in enumerate(vocab)}
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(word2idx, f, ensure_ascii=False, indent=4)
        print(f"詞彙表已儲存至 '{output_path}'\n總詞彙數：{len(word2idx)}")
        return word2idx

    def tokenize_sentence(self, sentence, word2idx):
        """將句子轉換為token ID序列"""
        cleaned = self.clean_sentence(sentence)
        tokens = ["<SOS>"] + list(cleaned) + ["<EOS>"]
        token_ids = [word2idx[token] for token in tokens if token in word2idx]
        return token_ids, len(token_ids)

    def find_max_sentence_length(self, sentences, word2idx):
        """找出最長句子長度"""
        max_length, longest_sentence = 0, ""
        for sentence in tqdm(sentences, desc="找出最長句子"):
            _, length = self.tokenize_sentence(sentence, word2idx)
            if length > max_length:
                max_length, longest_sentence = length, sentence
        return max_length, longest_sentence

    def batch_process_to_hdf5(self, audio_processor, audio_folder, df, output_path, workers=4):
        """批次處理音訊並儲存至HDF5"""
        with h5py.File(output_path, 'a') as h5f:
            existing_keys = set(h5f.keys())
            rows_to_process = [row for _, row in df.iterrows() if os.path.splitext(os.path.basename(row['path']))[0] not in existing_keys]
            print(f"總共 {len(df)} 筆資料，尚未處理 {len(rows_to_process)} 筆資料")
            def process_and_save(row):
                audio_path = os.path.join(audio_folder, row['path'])
                group_name = os.path.splitext(os.path.basename(audio_path))[0]
                audio_processor.save_to_hdf5(audio_path, h5f, group_name)
            with ThreadPoolExecutor(max_workers=workers) as executor:
                list(tqdm(executor.map(process_and_save, rows_to_process), total=len(rows_to_process), desc="處理並儲存至HDF5"))
        print(f"所有音訊已存入 {output_path}")