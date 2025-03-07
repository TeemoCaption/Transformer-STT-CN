import tensorflow as tf
from tensorflow.keras import layers, Model

# 定義 Transformer Encoder block
class TransformerEncoderBlock(layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        super(TransformerEncoderBlock, self).__init__()
        self.mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.dropout1 = layers.Dropout(dropout_rate)
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        
        self.ffn = tf.keras.Sequential([
            layers.Dense(dff, activation='relu'),
            layers.Dense(d_model)
        ])
        self.dropout2 = layers.Dropout(dropout_rate)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
    
    def call(self, x, training=False):
        attn_output = self.mha(x, x, x)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

# 定義 Transformer-CTC 模型
class TransformerCTC(Model):
    def __init__(self, audio_input_shape, vocab_size, d_model=256, 
                 num_enc_layers=6, num_heads=8, dff=512, dropout_rate=0.1):
        """
        參數:
          - audio_input_shape: 音訊特徵（例如 spectrogram）的輸入形狀
          - vocab_size: 標籤數量（不包含 CTC blank token）
          - d_model: 隱藏單元維度
          - num_enc_layers: Transformer Encoder 層數
          - num_heads: 多頭注意力的頭數
          - dff: 前饋網路中間層的單元數
          - dropout_rate: dropout 比例
        """
        super(TransformerCTC, self).__init__()
        self.audio_input_shape = audio_input_shape
        # CTC blank token 預留 0，因此最終輸出類別數為 vocab_size + 1
        self.num_classes = vocab_size + 1  
        self.d_model = d_model
        
        # 建立 CNN 前端：提取音訊特徵
        self.cnn_frontend = self.build_cnn_frontend()
        
        # Transformer Encoder 層堆疊
        self.encoder_layers = [
            TransformerEncoderBlock(d_model, num_heads, dff, dropout_rate) 
            for _ in range(num_enc_layers)
        ]
        self.dropout = layers.Dropout(dropout_rate)
        
        # 最後輸出 logits（不含 activation），形狀：(batch, time, num_classes)
        self.final_dense = layers.Dense(self.num_classes, activation=None)
    
    def build_cnn_frontend(self):
        """
        CNN 前端：將輸入的 spectrogram 經過卷積、批次正規化，
        重塑並投影到 d_model 維度。
        """
        input_layer = layers.Input(shape=self.audio_input_shape)
        # 擴展 channel 維度
        x = tf.expand_dims(input_layer, -1)
        x = layers.Conv2D(32, (3, 3), strides=(2, 2), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(64, (3, 3), strides=(2, 2), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        # 重新塑形，保留時間軸，將其他維度合併為 features
        x = layers.Reshape((-1, x.shape[2] * x.shape[3]))(x)
        x = layers.Dense(self.d_model)(x)
        return Model(inputs=input_layer, outputs=x)
    
    def positional_encoding(self, seq_len, d_model):
        """產生位置編碼"""
        pos = tf.range(seq_len, dtype=tf.float32)[:, tf.newaxis]
        i = tf.range(d_model, dtype=tf.float32)[tf.newaxis, :]
        angle_rates = 1 / tf.pow(10000.0, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        angle_rads = pos * angle_rates
        sines = tf.math.sin(angle_rads[:, 0::2])
        cosines = tf.math.cos(angle_rads[:, 1::2])
        pos_encoding = tf.concat([sines, cosines], axis=-1)
        return pos_encoding[tf.newaxis, ...]
    
    def call(self, inputs, training=False):
        """
        前向傳播：
         - inputs: 音訊 spectrogram，形狀為 (batch, height, width)
         - 輸出: logits，形狀為 (batch, time, num_classes)
        """
        x = self.cnn_frontend(inputs, training=training)
        seq_len = tf.shape(x)[1]
        pos_encoding = self.positional_encoding(seq_len, self.d_model)
        x = x + pos_encoding
        x = self.dropout(x, training=training)
        for encoder in self.encoder_layers:
            x = encoder(x, training=training)
        logits = self.final_dense(x)
        return logits

    def compute_ctc_loss(self, y_true, y_pred):
        batch_size = tf.shape(y_pred)[0]
        # Get the input sequence length
        input_length = tf.fill([batch_size], tf.shape(y_pred)[1])
        
        # Convert logits to log probabilities with softmax
        log_probs = tf.nn.log_softmax(y_pred, axis=-1)
        
        # Find label lengths by counting non-zero values
        label_length = tf.reduce_sum(tf.cast(tf.not_equal(y_true, 0), tf.int32), axis=1)
        
        # Ensure label_length is at least 1 to avoid empty labels
        label_length = tf.maximum(label_length, 1)
        
        # Use tf.nn.ctc_loss instead of keras.backend.ctc_batch_cost
        loss = tf.nn.ctc_loss(
            labels=tf.cast(y_true, tf.int32),
            logits=log_probs,
            label_length=label_length,
            logit_length=input_length,
            blank_index=0,
            logits_time_major=False
        )
        
        return tf.reduce_mean(loss)

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

    def get_config(self):
        return {"d_model": self.d_model.numpy(), "warmup_steps": self.warmup_steps}