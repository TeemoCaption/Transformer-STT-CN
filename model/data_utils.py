import os
import sys

def get_data(wavs, id_to_text, maxlen=50):
    """
    返回音頻路徑和轉錄文本的映射\n
    參數：\n
    id_to_text: 音頻id和轉錄文本的映射\n
    maxlen: 最大文本長度
    """
    data = []
    #print(maxlen)
    for w in wavs:
        id = os.path.basename(w).split(".")[0]
        # 如果文本長度小於 maxlen，則將音頻路徑和文本添加到 data 中
        if len(id_to_text[id]) < maxlen:
            data.append({"audio": w, "text": id_to_text[id]})

    return data

class VectorizeChar:
    """
    將文本轉換為數字序列（字符的索引值），並控制文本長度
    """
    def __init__(self, max_len=50):
        """
        參數：\n
        max_len: 最大文本長度
        """
        # 將英文字母（a-z）轉換為對應的字符，並加入一些特殊字符。
        self.vocab = (
            ["-","#" ,"<", ">"]
            + [chr(i + 96) for i in range(1, 27)]
            + [" ", ".", ",", "?"]
        )
        self.max_len = max_len
        self.char_to_idx = {}
        for i, ch in enumerate(self.vocab):
            self.char_to_idx[ch] = i

    # 將文本轉換為數字序列(當對象像函數一樣被調用時會執行這個方法)
    def __call__(self, text):
        """
        參數：\n
        text: 文本
        """
        text = text.lower()
        # 如果文本長度大於最大長度，則截斷文本
        text = text[: self.max_len - 2]
        text = "<" + text + ">"
        # 將序列長度填充到 max_len，缺少的部分用 0 填充
        pad_len = self.max_len - len(text)
        # 返回字符的索引值，如果字符在字典中找不到，則會用預設的 1 來處理，最後通過補充 0 來確保文本長度達到 max_len。
        return [
            self.char_to_idx.get(ch, 1) for ch in text
        ] + [0] * pad_len


    def get_vocabulary(self):
        return self.vocab