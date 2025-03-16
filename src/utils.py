import os
import pandas as pd
import subprocess
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc  # 垃圾回收模組
from multiprocessing import cpu_count

def convert_mp3_to_wav(input_path, output_path, sample_rate=16000):
    """
    使用 ffmpeg 將 mp3 轉換為 16kHz 單聲道的 wav 格式。
    如果輸出檔案已存在，則直接略過。
    """
    if os.path.exists(output_path):
        return "已存在，跳過"

    command = [
        'ffmpeg', '-y', '-i', input_path,
        '-ac', '1',
        '-ar', str(sample_rate),
        output_path
    ]
    
    result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    if result.returncode != 0:
        return f"轉換失敗: {input_path}"
    
    return "轉換成功"

def generate_tasks(tsv_path, clips_dir, output_dir):
    """
    逐行讀取 TSV，使用 generator 來減少記憶體消耗。
    現在會產生所有任務，讓 convert_mp3_to_wav 自行檢查檔案是否已存在，
    以便略過已處理過的檔案。
    """
    df_iter = pd.read_csv(tsv_path, sep='\t', low_memory=False, chunksize=1000)
    for df in df_iter:
        for _, row in df.iterrows():
            mp3_path = os.path.join(clips_dir, row['path'])
            relative_path = os.path.splitext(row['path'])[0] + '.wav'
            wav_path = os.path.join(output_dir, relative_path)
            os.makedirs(os.path.dirname(wav_path), exist_ok=True)
            yield (mp3_path, wav_path)

def process_tsv(tsv_path, clips_dir, output_dir):
    """
    讀取 TSV 檔案，逐個處理轉換任務，並略過已處理過的檔案，
    同時計算成功轉換、略過及失敗的數量。
    """
    print(f"讀取 {os.path.basename(tsv_path)} ...")
    num_workers = min(cpu_count(), 2)
    print(f"使用 {num_workers} 個執行緒進行轉換 ...")
    
    # 產生所有任務，不再過濾已存在的檔案
    tasks = list(generate_tasks(tsv_path, clips_dir, output_dir))
    converted, skipped, failed = 0, 0, 0

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(convert_mp3_to_wav, mp3, wav): (mp3, wav) for mp3, wav in tasks}

        for future in tqdm(as_completed(futures), total=len(futures), desc=f"轉換 {os.path.basename(tsv_path)}"):
            res = future.result()
            if res == "轉換成功":
                converted += 1
            elif res == "已存在，跳過":
                skipped += 1
            elif "轉換失敗" in res:
                failed += 1

            if (converted + skipped + failed) % 100 == 0:
                gc.collect()

    print(f"\n{os.path.basename(tsv_path)} 轉換結果:")
    print(f"成功轉換: {converted}")
    print(f"已存在，略過: {skipped}")
    if failed > 0:
        print(f"轉換失敗: {failed}")

    gc.collect()
