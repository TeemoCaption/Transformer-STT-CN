# main.py

import os
from src.utils import process_tsv

dataset_dir = 'dataset'
clips_dir = os.path.join(dataset_dir, 'clips')
output_dir = os.path.join(dataset_dir, 'processed')

tsv_files = [
    os.path.join(dataset_dir, 'train.tsv'),
    os.path.join(dataset_dir, 'validated.tsv'),
    os.path.join(dataset_dir, 'test.tsv')
]

if __name__ == '__main__':
    for tsv_file in tsv_files:
        print(f"開始處理 {os.path.basename(tsv_file)} ...")
        process_tsv(tsv_file, clips_dir, output_dir)
        print(f"{os.path.basename(tsv_file)} 處理完成。\n")
