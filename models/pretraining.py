import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np

# -------------------------------
# 1. 特徵擷取器：多層 1D 卷積
# -------------------------------
class FeatureEncoder(layers.Layer):
    def __init__(self, num_layers=7, hidden_size=512, **kwargs):
        super(FeatureEncoder, self).__init__(**kwargs)
        self.conv_layers = []
        for i in range(num_layers):
            self.conv_layers.append(
                layers.Conv1D(filters=hidden_size,
                              kernel_size=3,
                              strides=2,      # 每層下採樣，降低時間解析度
                              padding='same',
                              activation='relu')
            )
    
    def call(self, inputs):
        # 輸入 shape: (batch, time, channels)，假設 channels=1（單聲道）
        x = inputs
        for conv in self.conv_layers:
            x = conv(x)
        return x  # 輸出 shape: (batch, T', hidden_size)

# --------------------------------------------------
# 2. 連續遮罩模組：隨機遮罩部分連續時間步特徵
# --------------------------------------------------
class ContiguousMasking(layers.Layer):
    def __init__(self, mask_prob=0.065, mask_length=10, **kwargs):
        """
        mask_prob: 每個時間步被遮罩的機率（大約 6.5%，依原論文設定）
        mask_length: 每次遮罩的連續區塊長度（時間步數）
        """
        super(ContiguousMasking, self).__init__(**kwargs)
        self.mask_prob = mask_prob
        self.mask_length = mask_length
        # 我們不在 __init__ 中建立 mask_embedding，而是在 build() 中建立

    def build(self, input_shape):
        # input_shape: (batch, time, dim)
        self.feature_dim = input_shape[-1]
        # 在 build() 中根據 feature_dim 建立 mask_embedding，形狀為 (1, 1, feature_dim)
        self.mask_embedding = self.add_weight(
            name="mask_embedding", 
            shape=(1, 1, self.feature_dim),
            initializer=tf.keras.initializers.Zeros(),
            trainable=True
        )
        super(ContiguousMasking, self).build(input_shape)
    
    def call(self, inputs, training=False):
        if not training:
            return inputs

        batch_size = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1]
        num_mask = tf.cast(tf.math.floor(tf.cast(seq_len, tf.float32) * self.mask_prob / self.mask_length), tf.int32)

        # 定義一個只依賴於外部 seq_len 的 mask_sample 函數
        def mask_sample(_):
            mask = tf.zeros((seq_len,), dtype=tf.bool)
            max_start = seq_len - self.mask_length
            # 如果 max_start 不正確（例如太小），直接返回全 0 的 mask
            def true_fn():
                starts = tf.random.uniform([num_mask], minval=0, maxval=max_start, dtype=tf.int32)
                # 迭代所有隨機起點，更新 mask
                for start in tf.unstack(starts):
                    indices = tf.range(start, start + self.mask_length)
                    mask_updated = tf.ones([self.mask_length], dtype=tf.bool)
                    mask = tf.tensor_scatter_nd_update(mask, tf.expand_dims(indices, 1), mask_updated)
                return mask
            def false_fn():
                return mask
            mask = tf.cond(max_start > 0, true_fn, false_fn)
            return mask

        # 遍歷 batch_size 個樣本，對每個生成一個 mask (shape: (seq_len,))
        masks = tf.map_fn(mask_sample, tf.range(batch_size), fn_output_signature=tf.bool)
        masks = tf.expand_dims(masks, -1)  # (batch, seq_len, 1)
        masked_inputs = tf.where(masks, tf.broadcast_to(self.mask_embedding, tf.shape(inputs)), inputs)
        return masked_inputs

# -----------------------------------------------
# 3. Transformer 上下文網路：堆疊 Transformer 區塊
# -----------------------------------------------
class TransformerEncoderBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1, **kwargs):
        super(TransformerEncoderBlock, self).__init__(**kwargs)
        self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.dropout1 = layers.Dropout(dropout_rate)
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation='relu'),
            layers.Dense(embed_dim)
        ])
        self.dropout2 = layers.Dropout(dropout_rate)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
    
    def call(self, inputs, training=False):
        attn_output = self.attention(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.norm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.norm2(out1 + ffn_output)

class TransformerEncoder(layers.Layer):
    def __init__(self, num_layers, embed_dim, num_heads, ff_dim, dropout_rate=0.1, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.blocks = [TransformerEncoderBlock(embed_dim, num_heads, ff_dim, dropout_rate)
                       for _ in range(num_layers)]
    
    def call(self, inputs, training=False):
        x = inputs
        for block in self.blocks:
            x = block(x, training=training)
        return x

# ----------------------------------------------------
# 4. 量化模組：利用 Gumbel-Softmax 進行向量量化
# ----------------------------------------------------
def sample_gumbel(shape, eps=1e-20):
    U = tf.random.uniform(shape, minval=0, maxval=1)
    return -tf.math.log(-tf.math.log(U + eps) + eps)

def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(tf.shape(logits))
    return tf.nn.softmax(y / temperature)

class GumbelVectorQuantizer(layers.Layer):
    def __init__(self, codebook_size=320, embed_dim=512, temperature=0.5, **kwargs):
        """
        codebook_size: 碼本大小
        embed_dim: 與特徵維度一致
        temperature: Gumbel-Softmax 的溫度參數
        """
        super(GumbelVectorQuantizer, self).__init__(**kwargs)
        self.codebook_size = codebook_size
        self.embed_dim = embed_dim
        self.temperature = temperature

    def build(self, input_shape):
        # 初始化碼本，shape: (codebook_size, embed_dim)
        self.codebook = self.add_weight("codebook",
                                        shape=(self.codebook_size, self.embed_dim),
                                        initializer=tf.keras.initializers.RandomUniform(-1, 1),
                                        trainable=True)
        super(GumbelVectorQuantizer, self).build(input_shape)
    
    def call(self, inputs, training=False):
        # inputs: (batch, time, embed_dim)
        # 計算與每個碼本向量的相似度（內積），得到 logits: (batch, time, codebook_size)
        logits = tf.einsum('btd,vd->btv', inputs, self.codebook)
        # 使用 Gumbel-Softmax 取得近似 one-hot 編碼
        soft_one_hot = gumbel_softmax_sample(logits, self.temperature)
        # 量化：將 one-hot 與碼本做乘法，結果 shape 與 inputs 相同
        quantized = tf.einsum('btv,vd->btd', soft_one_hot, self.codebook)
        return quantized, soft_one_hot  # 同時返回量化結果及編碼分布

# ----------------------------------------------------
# 5. 對比損失函數（Contrastive Loss, InfoNCE Loss）
# ----------------------------------------------------
def contrastive_loss(context, quantized, mask, temperature=0.1):
    """
    計算對比損失：
      context: Transformer 輸出 (batch, time, dim)
      quantized: 量化模組目標 (batch, time, dim)
      mask: 布林遮罩，指示哪些時間步是被遮罩的（用於計算損失）
    此函數簡化處理：以每個被遮罩位置為正樣本，其他位置作為負樣本
    """
    # 將輸入攤平成 (B*T, dim)
    context_flat = tf.reshape(context, [-1, tf.shape(context)[-1]])
    quantized_flat = tf.reshape(quantized, [-1, tf.shape(quantized)[-1]])
    mask_flat = tf.reshape(mask, [-1])  # 布林值，True 表示該位置計算損失

    # 僅取被遮罩位置
    pos_context = tf.boolean_mask(context_flat, mask_flat)
    pos_quantized = tf.boolean_mask(quantized_flat, mask_flat)
    
    # 計算 cosine 相似度
    pos_context_norm = tf.math.l2_normalize(pos_context, axis=1)
    pos_quantized_norm = tf.math.l2_normalize(pos_quantized, axis=1)
    pos_sim = tf.reduce_sum(pos_context_norm * pos_quantized_norm, axis=1)  # (N,)

    # 作為負樣本：這裡以同一 batch 內所有未遮罩位置作負樣本（示範用途）
    neg_context = context_flat  # (B*T, dim)
    neg_context_norm = tf.math.l2_normalize(neg_context, axis=1)
    # 兩兩計算相似度矩陣 (N, B*T)
    sim_matrix = tf.matmul(pos_context_norm, neg_context_norm, transpose_b=True)
    # 提取正樣本分數：位於對角線位置（假設正樣本與自身比對）
    # 為簡化處理，這裡用 pos_sim 代替（實際上需要更複雜的負樣本抽樣）
    logits = sim_matrix / temperature
    # 建立標籤：對每個正樣本，理想情況下正樣本得分最高
    labels = tf.range(tf.shape(pos_context_norm)[0])
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    return tf.reduce_mean(loss)

# ----------------------------------------------------
# 6. 組合整體模型：Wav2Vec 2.0 模型
# ----------------------------------------------------
class Wav2Vec2Model(Model):
    def __init__(self,
                 cnn_layers=7,
                 cnn_hidden_size=512,
                 transformer_layers=12,
                 embed_dim=512,
                 num_heads=8,
                 ff_dim=2048,
                 codebook_size=320,
                 mask_prob=0.065,
                 mask_length=10,
                 dropout_rate=0.1,
                 **kwargs):
        super(Wav2Vec2Model, self).__init__(**kwargs)
        self.feature_encoder = FeatureEncoder(num_layers=cnn_layers, hidden_size=cnn_hidden_size)
        self.masking = ContiguousMasking(mask_prob=mask_prob, mask_length=mask_length)
        self.transformer = TransformerEncoder(num_layers=transformer_layers,
                                              embed_dim=embed_dim,
                                              num_heads=num_heads,
                                              ff_dim=ff_dim,
                                              dropout_rate=dropout_rate)
        self.quantizer = GumbelVectorQuantizer(codebook_size=codebook_size, embed_dim=embed_dim)
    
    def call(self, inputs, training=False):
        # 輸入: (batch, time, channels)；若原始 waveform 為 (batch, time)，需展開 channel 維度
        x = inputs
        if tf.rank(x) == 2:
            x = tf.expand_dims(x, -1)  # 轉為 (batch, time, 1)
        # 1. 特徵擷取器：CNN 將原始音訊轉換為低時序分辨率特徵
        features = self.feature_encoder(x)  # (batch, T', embed_dim)
        # 2. 對特徵進行連續遮罩（僅在訓練時使用）
        if training:
            masked_features = self.masking(features, training=training)
        else:
            masked_features = features
        # 3. Transformer 上下文網路
        context = self.transformer(masked_features, training=training)  # (batch, T', embed_dim)
        # 4. 量化：針對原始 CNN 特徵（未遮罩）進行離散化處理
        quantized, quantized_probs = self.quantizer(features, training=training)  # (batch, T', embed_dim)
        return context, quantized