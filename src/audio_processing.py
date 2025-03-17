import os
import subprocess
import contextlib
import wave
import collections
import webrtcvad
import tempfile

class Frame:
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration

def read_wave(path):
    """
    讀取 WAV 文件，並返回原始音頻數據與採樣率。
    要求：單聲道、16位 PCM、16kHz 取樣率。
    """
    with contextlib.closing(wave.open(path, 'rb')) as wf:
        channels = wf.getnchannels()
        if channels != 1:
            raise ValueError("僅支援單聲道音頻")
        sample_width = wf.getsampwidth()
        if sample_width != 2:
            raise ValueError("僅支援16位PCM格式")
        sample_rate = wf.getframerate()
        if sample_rate != 16000:
            raise ValueError("音頻需為16kHz")
        frames = wf.readframes(wf.getnframes())
    return frames, sample_rate

def frame_generator(frame_duration_ms, audio, sample_rate):
    """
    將音頻數據分割成固定時長的 frames。
    每個 frame 時長以毫秒計算，依據 16kHz 及 16 位，每個樣本 2 個位元組。
    """
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = frame_duration_ms / 1000.0
    while offset + n <= len(audio):
        yield Frame(audio[offset:offset + n], timestamp, duration)
        timestamp += duration
        offset += n

def vad_collector(sample_rate, frame_duration_ms, padding_duration_ms, vad, frames):
    """
    利用 VAD 判斷，收集所有語音段落（刪除靜音部分）。
    
    返回值是一個列表，每個元素為 (開始時間, 語音段落原始數據)。
    """
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    triggered = False
    voiced_frames = []
    segments = []
    for frame in frames:
        is_speech = vad.is_speech(frame.bytes, sample_rate)
        if not triggered:
            ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])
            if num_voiced > 0.9 * ring_buffer.maxlen:
                triggered = True
                voiced_frames.extend([f for f, s in ring_buffer])
                ring_buffer.clear()
        else:
            voiced_frames.append(frame)
            ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                triggered = False
                segment = b"".join([f.bytes for f in voiced_frames])
                segments.append((voiced_frames[0].timestamp, segment))
                ring_buffer.clear()
                voiced_frames = []
    if voiced_frames:
        segment = b"".join([f.bytes for f in voiced_frames])
        segments.append((voiced_frames[0].timestamp, segment))
    return segments

def process_audio(mp3_path, output_wav, sample_rate=16000, aggressiveness=3, frame_duration_ms=30, padding_duration_ms=300):
    """
    處理音訊檔案：
      1. 將 MP3 轉換為 16kHz 單聲道 WAV（暫存檔）。
      2. 利用 VAD 刪除靜音段，僅保留語音部分，並將語音段落串接成一個 WAV 檔案（output_wav）。
    
    參數：
      mp3_path: 輸入的 MP3 檔案路徑。
      output_wav: 輸出的最終 WAV 檔案路徑。
      sample_rate: 採樣率（預設 16000）。
      aggressiveness: VAD 靈敏度 (0~3，數值越大表示更嚴格)。
      frame_duration_ms: 每個 frame 的時長（毫秒），建議 20～30 ms。
      padding_duration_ms: 合併語音段時允許的靜音時長（毫秒）。
    
    回傳：
      成功訊息或含詳細錯誤訊息的提示。
    """
    if os.path.exists(output_wav):
        return "已存在，跳過"
    
    # 檢查輸入 MP3 是否存在
    if not os.path.exists(mp3_path):
        return f"檔案不存在: {mp3_path}"
    
    os.makedirs(os.path.dirname(output_wav), exist_ok=True)
    
    # 將 MP3 轉換成 WAV，存放到臨時檔案中
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
        temp_wav_path = temp_wav.name
        
    convert_cmd = [
        "ffmpeg", "-y", "-i", mp3_path,
        "-ac", "1",
        "-ar", str(sample_rate),
        temp_wav_path
    ]
    result = subprocess.run(convert_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        error_msg = result.stderr.decode('utf-8', errors='replace')
        os.remove(temp_wav_path)
        return f"轉換失敗: {mp3_path}, error: {error_msg}"
    
    try:
        audio, sr = read_wave(temp_wav_path)
    except Exception as e:
        os.remove(temp_wav_path)
        return f"讀取 WAV 失敗: {mp3_path}, error: {str(e)}"
    
    try:
        vad = webrtcvad.Vad(aggressiveness)
        frames = list(frame_generator(frame_duration_ms, audio, sr))
        segments = vad_collector(sr, frame_duration_ms, padding_duration_ms, vad, frames)
    except Exception as e:
        os.remove(temp_wav_path)
        return f"VAD 處理失敗: {mp3_path}, error: {str(e)}"
    
    # 將所有語音段落串接，達到刪除靜音的效果
    combined_audio = b"".join([segment for _, segment in segments])
    
    try:
        with wave.open(output_wav, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(combined_audio)
    except Exception as e:
        os.remove(temp_wav_path)
        return f"儲存輸出失敗: {mp3_path}, error: {str(e)}"
    
    os.remove(temp_wav_path)
    duration_sec = len(combined_audio) / (2 * sr)
    return f"處理成功: 輸出檔案 {output_wav}，總語音時長約 {duration_sec:.2f} 秒"
