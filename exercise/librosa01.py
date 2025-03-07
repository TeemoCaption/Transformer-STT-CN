import librosa
import matplotlib.pyplot as plt
import numpy as np

# 顯示波形圖
def showWave():
    plt.figure(figsize=(14, 5))
    librosa.display.waveshow(y, sr=sr)
    plt.title("音頻波形")
    plt.xlabel("時間 (秒)")
    plt.ylabel("振幅")
    plt.show()


# 特徵提取
# 短時傅立葉變換
def extractSTFT(y, sr):
    # 計算短時傅立葉變換(STFT)
    D = librosa.stft(y)
    # 將振幅轉換為分貝
    DB = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    plt.figure(figsize=(14, 5))
    librosa.display.specshow(DB, sr=sr, x_axis="time", y_axis="hz")
    plt.colorbar(format="%+2.0f dB")
    plt.title("頻譜圖")
    plt.show()


# 梅爾頻譜圖
def showMel(y, sr):
    # 生成梅爾頻譜圖
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_DB = librosa.power_to_db(S, ref=np.max)

    plt.figure(figsize=(14, 5))
    librosa.display.specshow(S_DB, sr=sr, x_axis="time", y_axis="mel")
    plt.colorbar(format="%+2.0f dB")
    plt.title("梅爾頻譜圖")
    plt.show()


# 提取梅爾頻率倒譜系數 (MFCC) 特徵
def extractMFCC(y, sr):
    # 提取 MFCC 特徵，n_mfcc 指定 MFCC 的數量（通常 13 到 20 之間）
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    plt.figure(figsize=(14, 5))
    librosa.display.specshow(mfccs, sr=sr, x_axis="time")
    plt.colorbar()
    plt.title("MFCC")
    plt.show()


# 節拍追蹤與節奏分析
def beat(y, sr):
    # 節拍追蹤
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    print("預估節奏 (BPM): {:.2f}".format(tempo[0]))

    # 將節拍幀轉換成時間戳
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    print("節拍時間戳:", beat_times)


def main():
    # 設定字體為支援中文的字體
    plt.rcParams["font.family"] = "Microsoft JhengHei"
    plt.rcParams["axes.unicode_minus"] = False  # 解決負號顯示問題

    # 讀取音頻文件，默認將音頻轉為單聲道並重採樣到 22050 Hz
    y, sr = librosa.load(
        "../zh-TW/clips/0a0ce0fd2fd6f0552371e2d88d8a56e147fa8a6726cbf1c340f40904c9897dccaab17481cda09ab77d7158d3816bff5af275fc0522fa846d8737c489349e69fa.mp3"
    )
    print("音頻資料 shape:", y.shape)
    print("採樣率:", sr)
    # showWave()
    # extractSTFT(y, sr)
    # showMel(y, sr)
    # extractMFCC(y, sr)
    # beat(y, sr)


if __name__ == "__main__":
    main()
