# model.py
    
import tensorflow as tf
from tensorflow.keras import layers, Model

class ConformerBlock(layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1, kernel_size=3):
        """Conformer模塊，結合FFN、自注意力與卷積"""
        super(ConformerBlock, self).__init__()
        self.ffn1 = tf.keras.Sequential([layers.Dense(dff, activation='relu'), layers.Dense(d_model)])
        self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.layernorm_conv = layers.LayerNormalization(epsilon=1e-6)
        self.pointwise_conv1 = layers.Conv1D(2 * d_model, kernel_size=1, padding='same')
        self.activation_conv1 = layers.Activation('gelu')
        self.depthwise_conv = layers.DepthwiseConv1D(kernel_size=kernel_size, padding='same')
        self.batchnorm_conv = layers.BatchNormalization()
        self.activation_conv2 = layers.Activation('swish')
        self.pointwise_conv2 = layers.Conv1D(d_model, kernel_size=1, padding='same')
        self.ffn2 = tf.keras.Sequential([layers.Dense(dff, activation='relu'), layers.Dense(d_model)])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm4 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)
        self.dropout_conv = layers.Dropout(dropout_rate)
        self.dropout3 = layers.Dropout(dropout_rate)

    def call(self, x, training=False):
        """前向傳播"""
        ffn1_output = self.ffn1(x)
        x = self.layernorm1(x + 0.5 * self.dropout1(ffn1_output, training=training))
        attn_output = self.attention(x, x)
        x = self.layernorm2(x + self.dropout2(attn_output, training=training))
        conv_input = self.layernorm_conv(x)
        conv_output = self.pointwise_conv1(conv_input)
        conv_output = self.activation_conv1(conv_output)
        conv_output = self.depthwise_conv(conv_output)
        conv_output = self.batchnorm_conv(conv_output, training=training)
        conv_output = self.activation_conv2(conv_output)
        conv_output = self.pointwise_conv2(conv_output)
        x = self.layernorm3(x + self.dropout_conv(conv_output, training=training))
        ffn2_output = self.ffn2(x)
        x = self.layernorm4(x + 0.5 * self.dropout3(ffn2_output, training=training))
        return x

class Seq2Seq(Model):
    def __init__(self, audio_input_shape, target_seq_len, vocab_size, d_model=256, num_enc_layers=6, num_dec_layers=6, num_heads=8, dff=512, dropout_rate=0.1):
        """初始化Seq2Seq模型"""
        super(Seq2Seq, self).__init__()
        self.audio_input_shape = audio_input_shape
        self.target_seq_len = target_seq_len
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_enc_layers = num_enc_layers
        self.num_dec_layers = num_dec_layers
        self.num_heads = num_heads
        self.dff = dff
        self.dropout_rate = dropout_rate
        self.transformer = self.build_transformer_model()

    def positional_encoding(self, seq_len, d_model):
        """生成位置編碼"""
        pos = tf.range(seq_len, dtype=tf.float32)[:, tf.newaxis]
        i = tf.range(d_model, dtype=tf.float32)[tf.newaxis, :]
        angle_rates = 1 / tf.pow(10000.0, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        angle_rads = pos * angle_rates
        pos_encoding = tf.concat([tf.math.sin(angle_rads[:, 0::2]), tf.math.cos(angle_rads[:, 1::2])], axis=-1)
        return pos_encoding[tf.newaxis, ...]

    def cnn_frontend(self, inputs):
        """CNN前端提取音訊特徵"""
        x = tf.expand_dims(inputs, axis=-1)
        x = layers.Conv2D(32, (3, 3), strides=(2, 2), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(64, (3, 3), strides=(2, 2), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Reshape((-1, x.shape[2] * x.shape[3]))(x)
        x = layers.Dense(self.d_model)(x)
        return x

    def decoder_layer(self, x, enc_output):
        """解碼器層"""
        look_ahead_mask = tf.linalg.band_part(tf.ones((tf.shape(x)[1], tf.shape(x)[1])), -1, 0)
        attn1 = layers.MultiHeadAttention(num_heads=self.num_heads, key_dim=self.d_model)(x, x, attention_mask=look_ahead_mask)
        attn1 = layers.Dropout(self.dropout_rate)(attn1)
        out1 = layers.LayerNormalization(epsilon=1e-6)(x + attn1)
        attn2 = layers.MultiHeadAttention(num_heads=self.num_heads, key_dim=self.d_model)(out1, enc_output)
        attn2 = layers.Dropout(self.dropout_rate)(attn2)
        out2 = layers.LayerNormalization(epsilon=1e-6)(out1 + attn2)
        ffn_output = layers.Dense(self.dff, activation='relu')(out2)
        ffn_output = layers.Dense(self.d_model)(ffn_output)
        ffn_output = layers.Dropout(self.dropout_rate)(ffn_output)
        return layers.LayerNormalization(epsilon=1e-6)(out2 + ffn_output)

    def build_transformer_model(self):
        """構建Transformer模型"""
        encoder_inputs = layers.Input(shape=self.audio_input_shape, name="encoder_inputs")
        cnn_features = self.cnn_frontend(encoder_inputs)
        enc_x = layers.Lambda(lambda x: x + self.positional_encoding(tf.shape(x)[1], self.d_model))(cnn_features)
        enc_x = layers.Dropout(self.dropout_rate)(enc_x)
        for _ in range(self.num_enc_layers):
            enc_x = ConformerBlock(self.d_model, self.num_heads, self.dff, self.dropout_rate)(enc_x)
        encoder_outputs = enc_x
        decoder_inputs = layers.Input(shape=(self.target_seq_len - 1,), name="decoder_inputs")
        dec_emb = layers.Embedding(self.vocab_size, self.d_model)(decoder_inputs)
        dec_x = layers.Lambda(lambda x: x + self.positional_encoding(self.target_seq_len - 1, self.d_model))(dec_emb)
        dec_x = layers.Dropout(self.dropout_rate)(dec_x)
        for _ in range(self.num_dec_layers):
            dec_x = self.decoder_layer(dec_x, encoder_outputs)
        outputs = layers.Dense(self.vocab_size, activation='softmax')(dec_x)
        return tf.keras.Model(inputs=[encoder_inputs, decoder_inputs], outputs=outputs)

    def call(self, inputs, training=False):
        """模型前向傳播"""
        return self.transformer(inputs, training=training)

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        """自定義學習率調度器"""
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
