import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
import logging
import tensorflow as tf
import gc
from tqdm import tqdm

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.ERROR)
plt.rcParams["font.family"] = "Microsoft JhengHei"
plt.rcParams["axes.unicode_minus"] = False

class AudioPreprocess:
    def __init__(self, target_sr=16000, save_folder='./features', n_fft=1024, hop_length=256, win_length=1024):
        """初始化音訊前處理類別"""
        self.target_sr = target_sr
        self.save_folder = save_folder
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        os.makedirs(save_folder, exist_ok=True)

    def resample_audio(self, y, sr):
        """將音訊重採樣至目標取樣率"""
        if sr != self.target_sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=self.target_sr)
            logging.debug(f"已重採樣至 {self.target_sr} Hz")
        return y, self.target_sr

    def get_spectrogram(self, y, sr):
        """計算STFT頻譜並轉換為dB刻度"""
        D = librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length)
        return librosa.amplitude_to_db(np.abs(D), ref=np.max)

    def plot_spectrogram(self, D_db, sr):
        """繪製頻譜圖以供視覺化檢查"""
        plt.figure(figsize=(12, 6))
        librosa.display.specshow(D_db, sr=sr, hop_length=self.hop_length, x_axis='time', y_axis='log')
        plt.colorbar(label='強度 (dB)')
        plt.title('音訊 STFT 頻譜圖')
        plt.xlabel('時間 (秒)')
        plt.ylabel('頻率 (Hz)')
        plt.grid(True)
        plt.show()

    def save_to_hdf5(self, audio_path, h5_file, group_name):
        """處理音訊並儲存至HDF5檔案"""
        try:
            y, sr = librosa.load(audio_path, sr=None)
            y, sr = self.resample_audio(y, sr)
            spectrogram = self.get_spectrogram(y, sr)
            if group_name not in h5_file:
                h5_file.create_dataset(group_name, data=spectrogram, compression='gzip')
                logging.debug(f"已儲存特徵至 HDF5: {group_name}")
            else:
                logging.debug(f"{group_name} 已存在，跳過儲存")
        except Exception as e:
            logging.error(f"處理失敗 {audio_path}: {e}")

    def create_chunked_dataset(self, h5_path, df, data_utils, word2idx, target_seq_len, audio_input_shape, batch_size=32, shuffle=True):
        """建立批次資料生成器"""
        def batch_generator():
            while True:
                indices = list(range(len(df)))
                if shuffle:
                    np.random.shuffle(indices)
                batch_samples = []
                with h5py.File(h5_path, 'r') as h5f:
                    for idx in indices:
                        row = df.iloc[idx]
                        group_name = os.path.splitext(os.path.basename(row['path']))[0]
                        if group_name in h5f:
                            spec = h5f[group_name][:]
                            desired_time = audio_input_shape[1]
                            if spec.shape[1] > desired_time:
                                spec = spec[:, :desired_time]
                            elif spec.shape[1] < desired_time:
                                spec = np.pad(spec, ((0, 0), (0, desired_time - spec.shape[1])), mode='constant')
                            token_ids, _ = data_utils.tokenize_sentence(row['sentence'], word2idx)
                            token_ids = token_ids[:target_seq_len] if len(token_ids) > target_seq_len else token_ids + [word2idx["<PAD>"]] * (target_seq_len - len(token_ids))
                            decoder_input = token_ids[:-1]
                            decoder_target = token_ids[1:]
                            batch_samples.append((spec, decoder_input, decoder_target))
                            if len(batch_samples) == batch_size:
                                spec_batch, decoder_inputs_batch, decoder_targets_batch = zip(*batch_samples)
                                yield ((np.array(spec_batch, dtype=np.float32), np.array(decoder_inputs_batch, dtype=np.int32)), np.array(decoder_targets_batch, dtype=np.int32))
                                batch_samples = []
                                gc.collect()
                    if batch_samples:
                        spec_batch, decoder_inputs_batch, decoder_targets_batch = zip(*batch_samples)
                        yield ((np.array(spec_batch, dtype=np.float32), np.array(decoder_inputs_batch, dtype=np.int32)), np.array(decoder_targets_batch, dtype=np.int32))
                        gc.collect()

        ds = tf.data.Dataset.from_generator(
            batch_generator,
            output_signature=(
                (tf.TensorSpec(shape=(None, *audio_input_shape), dtype=tf.float32),
                 tf.TensorSpec(shape=(None, target_seq_len - 1), dtype=tf.int32)),
                tf.TensorSpec(shape=(None, target_seq_len - 1), dtype=tf.int32)
            )
        ).prefetch(tf.data.AUTOTUNE)
        return ds