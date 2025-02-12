import os
import yaml
import tensorflow as tf
from model.model import Transformer
from components.data_preprocessing import DataPreprocessing
from entity.model_entity import CreateTensors
from model.data_utils import VectorizeChar
from model.utils import CustomSchedule, DisplayOutputs

class SpeechTrainer:
    def __init__(self, config_path="./configs/config.yaml"):
        """
        初始化 SpeechTrainer，從 YAML 設定檔讀取參數。
        """
        self.config = self.load_config(config_path)

        # 讀取數據相關設定
        self.tsv_path = self.config["data"]["tsv_path"]
        self.audio_folder = self.config["data"]["audio_folder"]
        self.test_size = self.config["data"]["test_size"]
        self.max_target_len = self.config["data"]["max_target_len"]

        # 讀取訓練相關設定
        self.batch_size = self.config["training"]["batch_size"]
        self.val_batch_size = self.config["training"]["val_batch_size"]
        self.epochs = self.config["training"]["epochs"]
        self.num_classes = self.config["training"]["num_classes"]

        # 讀取模型超參數
        self.model_params = self.config["model"]

        # 初始化變數
        self.vectorizer = None
        self.train_dataset = None
        self.val_dataset = None
        self.model = None

    def load_config(self, config_path):
        """
        讀取 YAML 設定檔
        """
        with open(config_path, "r", encoding="utf-8") as file:
            return yaml.safe_load(file)

    def load_data(self):
        """
        載入並分割資料集，初始化向量化器與數據集
        """
        print("載入資料集並分割中...")
        data_processor = DataPreprocessing(self.tsv_path, self.audio_folder, test_size=self.test_size)
        train_data, val_data = data_processor.split_data()  # 直接調用 split_data()

        # 創建詞彙表
        sentences = [d["sentence"] for d in train_data]
        self.vectorizer = VectorizeChar(sentences, max_len=self.max_target_len)
        print(self.vectorizer.get_vocabulary())

        # 轉換為 TensorFlow 數據集
        self.train_dataset = CreateTensors(train_data, self.vectorizer).create_tf_dataset(bs=self.batch_size)
        self.val_dataset = CreateTensors(val_data, self.vectorizer).create_tf_dataset(bs=self.val_batch_size)

        print(f"資料集處理完成！訓練數據: {len(train_data)} 筆, 驗證數據: {len(val_data)} 筆")

    def initialize_model(self):
        """
        初始化 Transformer 模型
        """
        print("🔹 初始化 Transformer 模型...")
        self.model = Transformer(
            num_hid=self.model_params["num_hid"],
            num_head=self.model_params["num_head"],
            num_feed_forward=self.model_params["num_feed_forward"],
            target_maxlen=self.max_target_len,
            num_layers_enc=self.model_params["num_layers_enc"],
            num_layers_dec=self.model_params["num_layers_dec"],
            num_classes=self.num_classes,
        )
        print("模型初始化完成！")

    def train_model(self):
        """
        訓練模型
        """
        print("開始訓練...")
        loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1)

        # 設定學習率排程
        learning_rate = CustomSchedule(
            init_lr=0.00001,
            lr_after_warmup=0.001,
            final_lr=0.00001,
            warmup_epochs=15,
            decay_epochs=40,
            steps_per_epoch=len(self.train_dataset),
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate)

        # 編譯模型
        self.model.compile(optimizer=optimizer, loss=loss_fn)

        # 設置 Callbacks
        batch = next(iter(self.train_dataset))
        display_cb = DisplayOutputs(batch, self.vectorizer.get_vocabulary(), target_start_token_idx=2, target_end_token_idx=3)
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

        # 訓練
        self.model.fit(
            self.train_dataset,
            validation_data=self.val_dataset,
            epochs=self.epochs,
            callbacks=[display_cb, early_stopping],
        )
        print("訓練完成！")

    def save_model(self, save_path="checkpoints/final_model.h5"):
        """
        儲存模型
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        self.model.save(save_path)
        print(f"模型已儲存至 {save_path}")

    def run(self):
        """
        執行完整流程
        """
        self.load_data()
        # self.initialize_model()
        # self.train_model()
        # self.save_model()
