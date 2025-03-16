# src/audio_processing.py

import os
import subprocess

def loudnorm_silenceremove(input_wav, output_wav, sample_rate=16000):
    """
    使用 ffmpeg 對 WAV 檔進行音量正規化 (loudnorm) 與靜音去除 (silenceremove)。

    範例參數說明：
      - loudnorm: I=-23, LRA=7, TP=-2 => 可視實際需求做調整。
      - silenceremove: start/stop_duration=0.5, threshold=-50dB => 持續 0.5 秒低於 -50dB 即視為靜音。

    若需求不同，可自行調整參數或增加兩階段 loudnorm。
    """
    if os.path.exists(output_wav):
        return "已存在，跳過"

    # 確保輸出資料夾存在
    os.makedirs(os.path.dirname(output_wav), exist_ok=True)

    filter_cmd = (
        "loudnorm=I=-23:LRA=7:TP=-2:print_format=summary,"
        "silenceremove=start_periods=1:start_duration=0.5:start_threshold=-50dB:"
        "stop_periods=1:stop_duration=0.5:stop_threshold=-50dB"
    )

    command = [
        "ffmpeg", "-y",
        "-i", input_wav,
        "-af", filter_cmd,
        "-ac", "1",
        "-ar", str(sample_rate),
        output_wav
    ]

    result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if result.returncode != 0:
        return f"處理失敗: {input_wav}"

    return "處理成功"
