import os
import jieba

def get_data(sentences, maxlen=50):
    """
    取得句子資料\n
    參數：\n
    - sentences: 包含 `client_id` 和 `sentence` 的 DataFrame\n
    - maxlen: 句子的最大長度\n
    回傳：
    - data: 包含 `text` 的字典列表\n
    """
    data = []
    for _, row in sentences.iterrows():
        text = row["sentence"]
        if len(text) < maxlen:
            data.append({"text": text})

    return data

class VectorizeChar:
    """
    將中文文本轉換為數字序列（字符索引值），並控制文本長度
    """
    def __init__(self, sentences, max_len=50):
        """
        參數：\n
        - sentences: 用於建立字典的句子列表\n
        - max_len: 最大文本長度\n
        """
        # 建立中文字典（根據資料集動態生成）
        self.vocab = self.build_vocab(sentences)
        self.max_len = max_len
        self.char_to_idx = {ch: i for i, ch in enumerate(self.vocab)}

    def build_vocab(self, sentences):
        """
        根據句子建立字典
        """
        vocab_set = set()
        for text in sentences["sentence"]:
            vocab_set.update(text)  # 直接使用字元級別的分割
        vocab_list = ["<PAD>", "<UNK>", "<SOS>", "<EOS>"] + sorted(vocab_set)
        return vocab_list

    def __call__(self, text):
        """
        轉換文本為索引序列\n
        參數：\n
        - text: 輸入的句子\n
        回傳：\n
        - 索引列表，長度固定為 max_len
        """
        # 加入起始與結束標記
        text = "<SOS>" + text[: self.max_len - 2] + "<EOS>"  
        pad_len = self.max_len - len(text)
        return [self.char_to_idx.get(ch, 1) for ch in text] + [0] * pad_len  # 1 = <UNK>

    def get_vocabulary(self):
        return self.vocab
