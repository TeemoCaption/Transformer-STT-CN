from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Processor
import collections

class Frame:
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration

def get_processor(tokenizer):
    """
    建立音訊特徵提取器並結合 tokenizer 成為處理器。
    """
    # 建立音訊特徵提取器 (normalize 波形)
    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, do_normalize=True)
    # 結合 tokenizer 和 feature_extractor 成一個處理器
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    return processor

def prepare_batch(batch, processor):
    """
    對單筆資料進行預處理：
    
    1. 音訊：提取 array 並送入處理器獲得 input_values（返回值是 list，需要取 [0]）
    2. 文字：由於我們的詞彙表是依據字元建立，所以直接逐字查表，若該字不存在則用 unk_token_id 取代。
       這樣可以確保每個 token 都在合法範圍內，避免出現超出 vocab_size 的情況。
    """
    # 確保句子為字符串
    assert isinstance(batch["sentence"], str), "句子必須是字符串"
    
    # 1. 音訊處理：提取 array 並送入處理器獲得 input_values
    audio = batch["audio"]
    inputs = processor(audio["array"], sampling_rate=audio["sampling_rate"], do_normalize=True)
    batch["input_values"] = inputs.input_values[0]
    batch["input_length"] = len(inputs.input_values[0])
    
    # 2. 文字處理：逐字查詞彙表
    vocab = processor.tokenizer.get_vocab()  # 取得字元到 token id 的映射字典
    vocab_size = processor.tokenizer.vocab_size
    unk_id = processor.tokenizer.unk_token_id
    
    # 針對每個字元，若不存在於詞彙表中則替換為 unk_id
    labels = [vocab.get(char, unk_id) for char in batch["sentence"]]
    batch["labels"] = labels
    batch["labels_length"] = len(labels)
    
    # 檢查所有標籤都在合法範圍內（0 ~ vocab_size-1）
    assert all(0 <= token < vocab_size for token in labels), f"Label IDs out of range (vocab_size={vocab_size}): {labels}"
    return batch

def frame_generator(frame_duration_ms, audio, sample_rate):
    """
    將音訊分割為幀，注意 audio 為 bytes，
    每個樣本 16-bit (2 位元組)，因此需要乘以 2
    """
    frame_size = int(sample_rate * frame_duration_ms / 1000) * 2
    offset = 0
    while offset + frame_size <= len(audio):
        yield audio[offset:offset + frame_size]
        offset += frame_size

def vad_collector(sample_rate, frame_duration_ms, padding_duration_ms, vad, frames):
    """
    使用 VAD 收集語音段。
    """
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    triggered = False
    voiced_frames = []
    for frame in frames:
        is_speech = vad.is_speech(frame, sample_rate)
        if not triggered:
            ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])
            if num_voiced > 0.9 * ring_buffer.maxlen:
                triggered = True
                for f, s in ring_buffer:
                    voiced_frames.append(f)
                ring_buffer.clear()
        else:
            voiced_frames.append(frame)
            ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                triggered = False
                yield b''.join(voiced_frames), len(voiced_frames)
                ring_buffer.clear()
                voiced_frames = []
    if voiced_frames:
        yield b''.join(voiced_frames), len(voiced_frames)
