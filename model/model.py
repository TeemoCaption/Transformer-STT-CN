<<<<<<< HEAD
# model.py
from glob import glob
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# 對文本進行嵌入處理
class TokenEmbedding(layers.Layer):
    def __init__(self, num_vocab=1000, maxlen=100, num_hid=64):
        """
        參數：\n
        num_vocab: 詞彙表大小\n
        maxlen: 輸入序列的最大長度\n
        num_hid: 詞嵌入的維度
        """
        super().__init__()
        # 建立詞嵌入層(輸入的維度為num_vocab，輸出的維度為num_hid)
        self.emb = tf.keras.layers.Embedding(num_vocab, num_hid)
        # 建立位置編碼層(輸入的維度為maxlen，輸出的維度為num_hid)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=num_hid)

    # 定義呼叫方法
    def call(self, x):
        """
        參數：\n
        x: 輸入序列
        """
        # 獲取張量 x 的最後一個維度的大小，表示序列的長度
        maxlen = tf.shape(x)[-1]
        # 將輸入序列 x 通過詞嵌入層和位置編碼層
        x = self.emb(x)
        # tf.range用來生成一個數字範圍，返回的是一個一維的張量
        # start表示起始值，limit表示結束值，delta表示步長
        positions = tf.range(start=0, limit=maxlen, delta=1)
        # 將生成的數字範圍通過位置編碼層
        positions = self.pos_emb(positions)
        # 將詞嵌入和位置編碼的結果相加作為最終的輸出
        return x + positions

# 對語音特徵進行嵌入處理
class SpeechFeatureEmbedding(layers.Layer):
    def __init__(self, num_hid=64, maxlen=100):
        """
        參數：\n
        num_hid: 隱藏層的維度\n
        maxlen: 輸入序列的最大長度
        """
        super().__init__()
        # 建立卷積層1
        self.conv1 = tf.keras.layers.Conv1D(
            num_hid, 1, strides=2, padding="same", activation="relu"
        )
        # 建立卷積層2
        self.conv2 = tf.keras.layers.Conv1D(
            num_hid, 1, strides=2, padding="same", activation="relu"
        )
        # 建立卷積層3
        self.conv3 = tf.keras.layers.Conv1D(
            num_hid, 1, strides=2, padding="same", activation="relu"
        )
        # 建立位置編碼層(輸入的維度為maxlen，輸出的維度為num_hid)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=num_hid)

    # 定義呼叫方法
    def call(self, x):
        """
        參數：\n
        x: 輸入序列
        """
        x = self.conv1(x)
        x = self.conv2(x)
        return self.conv3(x)

# 定義Transformer的編碼器
class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, num_heads, feed_forward_dim, rate=0.1):
        """
        參數：\n
        embed_dim: 詞嵌入的維度\n
        num_heads: 注意力機制的頭數\n
        feed_forward_dim: 前饋神經網路的隱藏層維度\n
        rate: dropout比率
        """
        super().__init__()
        # 建立多頭注意力機制層
        # num_heads表示注意力機制的頭數，key_dim表示每個頭的維度
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        # 前饋神經網路
        self.ffn = keras.Sequential(
            [
                layers.Dense(feed_forward_dim, activation="relu"),
                layers.Dense(embed_dim),
            ]
        )
        # LayerNormalization層，epsilon是一個很小的數，避免分母為0
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6, dtype="float32")
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6, dtype="float32")
        self.dropout = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    # 定義呼叫方法
    def call(self, inputs, training):
        """
        參數：\n
        inputs: 輸入序列\n
        training: 是否訓練
        """
        attn_output = self.att(inputs, inputs)
        out1 = self.layernorm1(tf.cast(inputs, tf.float32) + tf.cast(attn_output, tf.float32))
        out1 = self.dropout(out1, training=training)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

# 定義Transformer的解碼器
class TransformerDecoder(layers.Layer):
    def __init__(self, embed_dim, num_heads, feed_forward_dim, dropout_rate=0.1):
        """
        參數：\n
        embed_dim: 詞嵌入的維度\n
        num_heads: 注意力機制的頭數\n
        feed_forward_dim: 前饋神經網路的隱藏層維度\n
        dropout_rate: dropout比率
        """
        super().__init__()
        # layer normalization層
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6, dtype="float32")
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6, dtype="float32")
        self.layernorm3 = layers.LayerNormalization(epsilon=1e-6, dtype="float32")

        # 多頭注意力機制層
        self.self_att = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.enc_att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.self_dropout = layers.Dropout(0.5)
        self.enc_dropout = layers.Dropout(0.1)
        self.ffn_dropout = layers.Dropout(0.1)
        # 前饋神經網路
        self.ffn = keras.Sequential(
            [
                layers.Dense(feed_forward_dim, activation="relu"),
                layers.Dense(embed_dim),
            ]
        )

    def causal_attention_mask(self, batch_size, n_dest, n_src, dtype):
        """
        遮蔽自注意力中的點積矩陣的上半部分。
        這樣可以防止未來的 token 影響當前的 token。
        在下三角形中標註 1，從右下角開始計算。\n
        參數：\n
        batch_size: 批次大小\n
        n_dest: 目標序列的長度\n
        n_src: 輸入序列的長度\n
        dtype: 資料類型
        """
        # [:, None] 是在原張量的每一個元素後加上一個新的維度，將其從一維變成了列向量
        i = tf.range(n_dest)[:, None]
        j = tf.range(n_src)
        m = i >= j
        # tf.cast是將張量 m 的數據類型轉換為指定的 dtype 類型
        mask = tf.cast(m, dtype)
        # tf.reshape是將張量 mask 重新塑形成指定的形狀
        mask = tf.reshape(mask, [1, n_dest, n_src])
        # tf.concat是將兩個張量拼接在一起
        # 0 代表沿著第一個維度（即行）進行拼接
        mult = tf.concat(
            [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)], 0
        )
        # tf.tile是將一個張量拓展成多個張量
        # mask：要重複的張量。mult：一個列表或張量，指定每個維度上應該重複的次數。
        return tf.tile(mask, mult)

    # 定義呼叫方法
    def call(self, enc_out, target):
        """
        參數：\n
        enc_out: 編碼器的輸出\n
        target: 解碼器的輸入
        """
        input_shape = tf.shape(target)
        # 獲取批次大小
        batch_size = input_shape[0]
        # 獲取序列長度
        seq_len = input_shape[1]
        # 生成目標序列的遮蔽
        causal_mask = self.causal_attention_mask(batch_size, seq_len, seq_len, tf.bool)
        target_att = self.self_att(target, target, attention_mask=causal_mask)

        target_norm = self.layernorm1(tf.cast(target, tf.float32) + self.self_dropout(target_att))

        enc_out = self.enc_att(target_norm, enc_out)
        enc_out_norm = self.layernorm2(self.enc_dropout(enc_out) + target_norm)

        ffn_out = self.ffn(enc_out_norm)
        ffn_out_norm = self.layernorm3(self.ffn_dropout(ffn_out) + enc_out_norm)

        return ffn_out_norm

# 定義Transformer模型
class Transformer(keras.Model):
    def __init__(
        self,
        num_hid=64,
        num_head=2,
        num_feed_forward=128,
        source_maxlen=100,
        target_maxlen=100,
        num_layers_enc=4,
        num_layers_dec=1,
        num_classes=10,
    ):
        """
        參數：\n
        num_hid: 隱藏層的維度\n
        num_head: 注意力機制的頭數\n
        num_feed_forward: 前饋神經網路的隱藏層維度\n
        source_maxlen: 輸入序列的最大長度\n
        target_maxlen: 輸出序列的最大長度\n
        num_layers_enc: 編碼器的層數\n
        num_layers_dec: 解碼器的層數\n
        num_classes: 詞彙表的大小
        """
        super().__init__()

        # 在每次批次訓練後更新計算出的平均值，並且在每次訓練步驟完成時返回這個平均值
        self.loss_metric = keras.metrics.Mean(name="loss")
        self.num_layers_enc = num_layers_enc
        self.num_layers_dec = num_layers_dec
        self.target_maxlen = target_maxlen
        self.num_classes = num_classes

        # 編碼器輸入
        self.enc_input = SpeechFeatureEmbedding(num_hid=num_hid, maxlen=source_maxlen)
        # 解碼器輸入
        self.dec_input = TokenEmbedding(
            num_vocab=num_classes, maxlen=target_maxlen, num_hid=num_hid
        )

        # 編碼器
        self.encoder = keras.Sequential(
            [self.enc_input]
            + [
                TransformerEncoder(num_hid, num_head, num_feed_forward)
                for _ in range(num_layers_enc)
            ]
        )

        # 解碼器
        for i in range(num_layers_dec):
            # setattr 是 Python 的內建函數，用來動態地設置對象的屬性
            # object：要設置屬性的對象。name：要設置的屬性名。value：要設置的屬性值。
            setattr(
                self,
                f"dec_layer_{i}",
                TransformerDecoder(num_hid, num_head, num_feed_forward),
            )
        # 分類器
        self.classifier = layers.Dense(num_classes)

    # 定義解碼方法
    def decode(self, enc_out, target):
        """
        參數：\n
        enc_out: 編碼器的輸出\n
        target: 解碼器的輸入
        """
        y = self.dec_input(target)
        for i in range(self.num_layers_dec):
            # getattr 是 Python 的內建函數，用來動態地獲取對象的屬性
            # object：要獲取屬性的對象。name：要獲取的屬性名。
            y = getattr(self, f"dec_layer_{i}")(enc_out, y)
        return y

    # 定義呼叫方法
    def call(self, inputs):
        """
        參數：\n
        inputs: 輸入序列
        """
        # 獲取輸入序列
        source = inputs[0]
        # 獲取目標序列
        target = inputs[1]
        enc_out = self.encoder(source)
        dec_out = self.decode(enc_out, target)
        y = self.classifier(dec_out)
        return y

    @property
    def metrics(self):
        return [self.loss_metric]

    # 訓練過程中處理每個批次的數據，計算損失，更新模型的權重，以及返回當前的損失值
    def train_step(self, batch):
        """
        參數：\n
        batch: 一個批次的數據
        """
        source = batch["source"]
        target = batch["target"]
        dec_input = target[:, :-1]
        dec_target = target[:, 1:]

        # 自動微分(梯度下降)-梯度磁帶
        with tf.GradientTape() as tape:
            preds = self([source, dec_input])
            # 將整數標籤轉換為 one-hot 編碼
            # depth：one-hot 編碼的維度
            one_hot = tf.one_hot(dec_target, depth=self.num_classes)
            # tf.math.equal(a, b) 是用來比較 a 和 b 是否相等的函數
            # tf.math.logical_not 是對張量進行邏輯取反
            mask = tf.math.logical_not(tf.math.equal(dec_target, 0))
            # self.compiled_loss用來計算損失
            # sample_weight 是樣本權重 sample_weight=mask，用來對計算損失時進行加權，用來忽略填充部分（即遮蔽部分）對損失的影響
            loss = self.compiled_loss(one_hot, preds, sample_weight=mask)

        # 計算梯度
        gradients = tape.gradient(loss, self.trainable_variables)
        # 更新權重
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        # 更新損失
        self.loss_metric.update_state(loss)
        return {"loss": self.loss_metric.result()}

    # 用來計算模型在驗證數據上的損失
    def val_loss(self, batch):
        source = batch["source"]
        target = batch["target"]
        # 獲取目標序列的輸入和輸出
        dec_input = target[:, :-1]
        # 獲取目標序列的輸出
        dec_target = target[:, 1:]

        preds = self([source, dec_input])
        one_hot = tf.one_hot(dec_target, depth=self.num_classes)
        mask = tf.math.logical_not(tf.math.equal(dec_target, 0))
        # 計算損失
        loss = self.compiled_loss(one_hot, preds, sample_weight=mask)
        return loss

    # 模型測試階段計算每個批次的損失並更新損失度量
    def test_step(self, batch):
        source = batch["source"]
        target = batch["target"]
        dec_input = target[:, :-1]
        dec_target = target[:, 1:]

        preds = self([source, dec_input])
        one_hot = tf.one_hot(dec_target, depth=self.num_classes)
        mask = tf.math.logical_not(tf.math.equal(dec_target, 0))
        loss = self.compiled_loss(one_hot, preds, sample_weight=mask)
        self.loss_metric.update_state(loss)
        return {"loss": self.loss_metric.result()}

    # 定義生成方法
    def generate(self, source, target_start_token_idx):
        """
        參數：\n
        source: 輸入序列\n
        target_start_token_idx: 目標序列的起始標記索引
        """
        bs = tf.shape(source)[0]
        enc = self.encoder(source)
        # tf.ones用來生成全為1的張量
        # bs 代表批次大小 (batch size)。1 代表初始化的輸入序列長度為 1（解碼器從一個起始標記開始生成）
        dec_input = tf.ones((bs, 1), dtype=tf.int32) * target_start_token_idx
        dec_logits = []

        for i in range(self.target_maxlen - 1):
            dec_out = self.decode(enc, dec_input)
            logits = self.classifier(dec_out)
            logits = tf.argmax(logits, axis=-1, output_type=tf.int32)
            last_logit = tf.expand_dims(logits[:, -1], axis=-1)
            dec_logits.append(last_logit)
            dec_input = tf.concat([dec_input, last_logit], axis=-1)

        return dec_input
=======
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
>>>>>>> aee2d04 (update)
