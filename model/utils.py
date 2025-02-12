import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# 顯示模型的輸出
class DisplayOutputs(keras.callbacks.Callback):
    """
    Keras Callback 裡，當這個 callback 被加入到 model.fit() 時，Keras 會自動把它的 model 屬性指向正在訓練的模型。
    """
    def __init__(
        self, batch, idx2token, target_start_token_idx=27, target_end_token_idx=28 
    ):
        """
        參數：\n
        batch: 一個批次的數據\n
        idx2token: 索引到標記的映射\n
        target_start_token_idx: 目標序列的起始標記索引\n
        target_end_token_idx: 目標序列的結束標記索引
        """
        self.batch = batch
        self.target_start_token_idx = target_start_token_idx
        self.target_end_token_idx = target_end_token_idx
        self.idx_to_char = idx2token
        
    # 在每個訓練時期結束時調用
    def on_epoch_end(self, epoch, logs=None):
        """
        參數：\n
        epoch: 訓練時期\n
        logs: 包含模型在該 epoch 中的評估指標（例如損失值）。
        """
        if epoch % 5 != 0:
            return
        source = self.batch["source"]
        target = self.batch["target"].numpy()
        # 計算當前批次的大小（即有多少條數據）
        bs = tf.shape(source)[0]
        # 生成目標序列
        preds = self.model.generate(source, self.target_start_token_idx)
        preds = preds.numpy()

        # 逐條數據進行預測
        for i in range(bs):
            # 獲取目標序列的文本
            # target[i, :]：取得批次中第 i 個樣本的目標序列 (target 是數值表示的序列)。
            # self.idx2token[_]：將數值表示的序列轉換為文本序列。
            target_text = "".join([self.idx_to_char[_] for _ in target[i, :]])
            prediction = ""
            # 獲取預測序列的文本
            for idx in preds[i, :]:
                prediction += self.idx_to_char[idx]
                # 如果預測到了結束標記，則停止生成
                if idx == self.target_end_token_idx:
                    break
            
            print(f"target: {target_text.replace('-', '')}")
            print(f"prediction: {prediction}\n")

# 自訂學習率排程            
class CustomSchedule(keras.optimizers.schedules.LearningRateSchedule):
    """
    自訂學習率排程\n
    參數：\n
    Warm-up 階段：學習率從 init_lr 緩慢增長到 lr_after_warmup。\n
    Decay 階段：學習率從 lr_after_warmup 線性減小到 final_lr。 
    """
    def __init__(
        self,
        init_lr=0.0001,
        lr_after_warmup=0.001,
        final_lr=0.00001,
        warmup_epochs=15,
        decay_epochs=85,
        steps_per_epoch=203,
    ):
        """
        參數:\n
        init_lr: 初始學習率\n
        lr_after_warmup: 熱身階段後的學習率\n
        final_lr: 最終學習率\n
        warmup_epochs: 熱身階段的epoch數\n
        decay_epochs: 衰減階段的epoch數\n
        steps_per_epoch: 每個epoch的步數
        """
        super().__init__()
        self.init_lr = init_lr
        self.lr_after_warmup = lr_after_warmup
        self.final_lr = final_lr
        self.warmup_epochs = warmup_epochs
        self.decay_epochs = decay_epochs
        self.steps_per_epoch = steps_per_epoch

    def calculate_lr(self, epoch):
        """
        線性熱身+線性衰減
        """
        # 初始學習率 + ((熱身階段後的學習率 - 初始學習率) / (熱身階段 - 1)) * epoch
        warmup_lr = (
            self.init_lr
            + ((self.lr_after_warmup - self.init_lr) / (self.warmup_epochs - 1)) * epoch
        )
        
        # 如果 epoch 小於熱身階段，則返回熱身學習率
        # tf.math.maximum 是取兩個數的最大值
        # 熱身階段後的學習率 + ((最終學習率 - 熱身階段後的學習率) / (衰減階段)) * (epoch - 熱身階段)
        decay_lr = tf.math.maximum(
            self.final_lr,
            self.lr_after_warmup
            * (epoch - self.warmup_epochs)
            / (self.decay_epochs) * (self.lr_after_warmup - self.final_lr),
        )

        return tf.math.minimum(warmup_lr, decay_lr)
    
    def __call__(self, step):
        """
        參數：\n
        step: 步數
        """
        epoch = step // self.steps_per_epoch
        return self.calculate_lr(epoch)

