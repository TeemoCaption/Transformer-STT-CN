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
    
    1. 音訊：提取 array 並送入處理器獲得 input_values （返回值是 list，需要取 [0]）
    2. 文字：使用 tokenizer 編碼轉錄為 labels 列表（不加特殊符號）
    """
    # 檢查句子是否為字符串
    assert isinstance(batch["sentence"], str), "句子必須是字符串"
    
    # 1. 音訊：提取 array 並送入處理器獲得 input_values （返回值是 list，需要取 [0]）
    audio = batch["audio"]
    inputs = processor(audio["array"], sampling_rate=audio["sampling_rate"], do_normalize=True)
    batch["input_values"] = inputs.input_values[0]
    # 記錄模型輸入長度（音訊樣本數）
    batch["input_length"] = len(inputs.input_values[0])
    
    # 2. 文字：使用 tokenizer 編碼轉錄為 labels 列表（不加特殊符號）
    labels = processor.tokenizer(batch["sentence"]).input_ids
    batch["labels"] = labels
    
    # 檢查標籤 ID 是否在詞彙大小範圍內
    max_id = max(labels) if labels else -1
    print(f"Sample sentence: '{batch['sentence']}', Labels: {labels}, Max ID: {max_id}")
    
    # 確保所有 ID 都在 0 到 vocab_size-1 之間
    vocab_size = processor.tokenizer.vocab_size
    assert all(0 <= id < vocab_size for id in labels), f"Label IDs out of range (vocab_size={vocab_size}): {labels}"
    
    # 記錄標籤長度（去除 -100 的有效標記數，在這裡就是序列長度）
    batch["labels_length"] = len(batch["labels"])
    return batch

def frame_generator(frame_duration_ms, audio, sample_rate):
    """
    將音訊分割為幀。
    """
    frame_size = int(sample_rate * frame_duration_ms / 1000)
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