import sys
import os
import tensorflow as tf
from model.model import Transformer
from components.data_preprocessing import DataPreprocessing
from entity.model_entity import CreateTensors
from model.data_utils import VectorizeChar
from model.utils import CustomSchedule, DisplayOutputs

# 設定參數
TSV_PATH = "./dataset/train.tsv"
AUDIO_FOLDER = "./dataset/clips/"     # 音檔資料夾

max_target_len = 50     # 資料中的所有轉錄文本小於 50 個字符
EPOCHS = 10                         # 訓練的總 Epoch 數
TEST_SIZE = 0.2                      # 驗證集比例

# 載入資料集並分割
sentences = [] 
data_processor = DataPreprocessing(TSV_PATH, AUDIO_FOLDER, test_size=TEST_SIZE)
train_data, val_data = data_processor.split_data()

# 建立詞彙表
vectorizer = VectorizeChar(sentences, max_len=max_target_len)
print("Vocab size", len(vectorizer.get_vocabulary()))

# 訓練集
train_tensor_creator = CreateTensors(train_data, vectorizer)
train_dataset = train_tensor_creator.create_tf_dataset(bs=16)

# 驗證集
val_tensor_creator = CreateTensors(val_data, vectorizer)
val_dataset = val_tensor_creator.create_tf_dataset(bs=4)