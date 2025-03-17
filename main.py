import os
from src.utils import process_tsv
import subprocess

# 設定資料夾路徑
dataset_dir = 'dataset'
clips_dir = os.path.join(dataset_dir, 'clips')
output_dir = os.path.join(dataset_dir, 'processed')

# TSV 檔案列表
tsv_files = [
    os.path.join(dataset_dir, 'train.tsv'),
    os.path.join(dataset_dir, 'validated.tsv'),
    os.path.join(dataset_dir, 'test.tsv')
]

def concatenate_processed_wavs(output_dir, final_output):
    """
    讀取 output_dir 中所有處理後的 WAV 檔（依路徑排序），
    並利用 ffmpeg 將它們串連成一個 final_output 檔案。
    """
    wav_files = []
    for root, _, files in os.walk(output_dir):
        for file in sorted(files):
            if file.endswith('.wav'):
                wav_files.append(os.path.join(root, file))
    
    if not wav_files:
        print("找不到任何處理後的 WAV 檔案，無法串連。")
        return

    # 建立供 ffmpeg concat 使用的清單檔案
    list_file = os.path.join(output_dir, "wav_list.txt")
    with open(list_file, "w", encoding="utf-8") as f:
        for wav in wav_files:
            # ffmpeg concat 要求每行以 file '完整路徑' 格式書寫
            f.write(f"file '{os.path.abspath(wav)}'\n")
    
    # 利用 ffmpeg concat demuxer 串連音訊（此處使用 copy 模式，不做 re-encode）
    concat_cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", list_file,
        "-c", "copy",
        final_output
    ]
    result = subprocess.run(concat_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        error_msg = result.stderr.decode('utf-8', errors='replace')
        print(f"串連失敗，錯誤訊息：{error_msg}")
    else:
        print(f"所有處理後的檔案已串連成 {final_output}")

if __name__ == '__main__':
    # 逐個 TSV 進行處理
    for tsv_file in tsv_files:
        print(f"開始處理 {os.path.basename(tsv_file)} ...")
        process_tsv(tsv_file, clips_dir, output_dir)
        print(f"{os.path.basename(tsv_file)} 處理完成。\n")
    
