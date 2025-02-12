import tensorflow as tf
from model import Transformer
from data_preprocessing import DataPreprocessing
from entity.model_entity import CreateTensors
from model.data_utils import VectorizeChar
from model.utils import CustomSchedule, DisplayOutputs

# 設定參數
TSV_PATH = "./dataset/train.tsv"
AUDIO_FOLDER = "./dataset/clips"     # 音檔資料夾
BATCH_SIZE = 4                      # 訓練批次大小
EPOCHS = 10                         # 訓練的總 Epoch 數
NUM_CLASSES = 10                     # 目標分類數
TEST_SIZE = 0.2                      # 驗證集比例

# 1. **載入數據並分割**
print("Loading dataset and splitting into train/val...")
sentences = []  # 你需要提供用於建立詞彙表的句子列表
data_processor = DataPreprocessing(TSV_PATH, AUDIO_FOLDER, sentences, test_size=TEST_SIZE)
train_data, val_data = data_processor.split_data()

# 2. **建立數據集**
vectorizer = VectorizeChar(sentences)

# 訓練集
train_tensor_creator = CreateTensors(train_data, vectorizer)
train_dataset = train_tensor_creator.create_tf_dataset(bs=BATCH_SIZE)

# 驗證集
val_tensor_creator = CreateTensors(val_data, vectorizer)
val_dataset = val_tensor_creator.create_tf_dataset(bs=BATCH_SIZE)

# 3. **初始化 Transformer 模型**
print("Initializing Transformer model...")
model = Transformer(num_classes=NUM_CLASSES)

# 4. **設定優化器與損失函數**
learning_rate = CustomSchedule()
optimizer = tf.keras.optimizers.Adam(learning_rate)
loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

# 5. **編譯模型**
model.compile(optimizer=optimizer, loss=loss_fn)

# 6. **設定回呼函數 (Callbacks)**
callbacks = [
    DisplayOutputs(batch=next(iter(train_dataset)), idx2token=vectorizer.get_vocabulary()),
    tf.keras.callbacks.ModelCheckpoint(filepath="checkpoints/model_{epoch}.h5", save_best_only=True)
]

# 7. **開始訓練 (加上驗證數據)**
print("Starting training...")
model.fit(train_dataset, validation_data=val_dataset, epochs=EPOCHS, callbacks=callbacks)

# 8. **保存最終模型**
model.save("final_model.h5")
print("Training completed and model saved.")
