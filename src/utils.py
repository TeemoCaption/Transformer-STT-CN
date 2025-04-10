from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Processor

class Frame:
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration

def get_processor(tokenizer):
    """
    建立音訊特徵提取器並結合 tokenizer 成為處理器。
    """
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1, 
        sampling_rate=16000,
        do_normalize=True
    )
    processor = Wav2Vec2Processor(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer
    )
    return processor

def prepare_batch(batch, processor):
    """
    對單筆資料進行預處理：
      1. 使用 processor 特徵提取器，取得 input_values
      2. 文字 -> token IDs
    """
    assert isinstance(batch["sentence"], str), "句子必須是字符串"

    # 音訊處理
    audio = batch["audio"]
    inputs = processor(audio["array"], 
                       sampling_rate=audio["sampling_rate"],
                       do_normalize=True)
    batch["input_values"] = inputs.input_values[0]
    batch["input_length"] = len(inputs.input_values[0])

    # 文字轉 label IDs
    vocab = processor.tokenizer.get_vocab()
    vocab_size = processor.tokenizer.vocab_size
    unk_id = processor.tokenizer.unk_token_id
    labels = [vocab.get(char, unk_id) for char in batch["sentence"]]

    # 這裡純粹用 assert 做二次保險檢查
    # 如果有 token ID 超過 vocab_size，就會噴錯
    assert all(0 <= t < vocab_size for t in labels), (
        f"Label IDs out of range (vocab_size={vocab_size}): {labels}"
    )
    batch["labels"] = labels
    batch["labels_length"] = len(labels)
    return batch
