import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc
from tqdm import tqdm  
from src.audio_processing import process_audio

def generate_tasks(tsv_path, clips_dir, output_dir):
    """
    逐行讀取 TSV，使用 generator 減少記憶體消耗，生成轉換任務。
    將每個輸出的 WAV 檔直接存放於 output_dir，不建立子資料夾。
    """
    df_iter = pd.read_csv(tsv_path, sep='\t', low_memory=False, chunksize=1000)
    for df in df_iter:
        for _, row in df.iterrows():
            mp3_path = os.path.join(clips_dir, row['path'])
            # 只取檔名，不保留原本的資料夾結構
            wav_filename = os.path.splitext(os.path.basename(row['path']))[0] + '.wav'
            wav_path = os.path.join(output_dir, wav_filename)
            yield (mp3_path, wav_path)

def process_tsv(tsv_path, clips_dir, output_dir):
    """
    讀取 TSV 檔案，處理 MP3 轉 WAV，並顯示處理進度條。
    """
    print(f"讀取 {os.path.basename(tsv_path)} ...")
    num_workers = 2  # 限制為 2 個執行緒
    print(f"使用 {num_workers} 個執行緒進行轉換 ...")
    os.makedirs(output_dir, exist_ok=True)
    # 生成任務列表
    tasks = list(generate_tasks(tsv_path, clips_dir, output_dir))
    total_tasks = len(tasks)
    processed, skipped, failed = 0, 0, 0

    # 使用 ThreadPoolExecutor 並顯示進度條
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        with tqdm(total=total_tasks, desc="處理進度") as pbar:
            futures = {executor.submit(process_audio, mp3, wav): (mp3, wav) for mp3, wav in tasks}

            for future in as_completed(futures):
                mp3, wav = futures[future]
                res = future.result().strip()  # 先 strip() 去除多餘空白與換行

                if res.startswith("處理成功"):
                    processed += 1
                elif res == "已存在，跳過":
                    skipped += 1
                else:
                    failed += 1
                    print(f"處理失敗: {mp3} - {res}")

                pbar.update(1)
                if (processed + skipped + failed) % 50 == 0:
                    gc.collect()

    print(f"\n{os.path.basename(tsv_path)} 處理結果:")
    print(f"成功處理: {processed}")
    print(f"已存在，略過: {skipped}")
    if failed > 0:
        print(f"處理失敗: {failed}")
    gc.collect()
