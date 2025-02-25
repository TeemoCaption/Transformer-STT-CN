# 語音轉文字專案 (Speech2Text Project)

這是一個簡單的語音轉文字 (Speech-to-Text, STT) 實作專案，使用 Python 開發，並基於 Transformer 模型來進行語音識別。

## 🚀 專案介紹
本專案的目標是將語音訊號轉換為對應的文字，並透過機器學習技術來提升準確度。

## 📂 資料集
本專案使用 ... 作為語音訓練資料集。
- 下載連結：[待更新]

## 📁 專案目錄結構

```
/Transformer-STT-CN
│── main.py               # 主程式
│── model/                # 模型相關程式碼
│   │── model.py          # Transformer 語音轉文字模型
│   │── utils.py          # 學習率調整 & 訓練過程顯示
|   |── data_utils.py     # 文本轉數字序列
│── entity/               # 創建資料集
|── |──model_entity.py 
│── cache/                # 快取資料夾
|── |──spectrogram_cache  # 離線儲存前處理頻譜（.npy 檔）
│── configs/              # 訓練參數、超參數設定
│   │── config.yaml       # 訓練設定
│── dataset/              # 存放數據集
│── checkpoints/          # 存放模型檢查點
│── requirements.txt      # 依賴套件
│── README.md             # 專案說明
│── .gitignore            # Git 忽略規則
```

## 🔧 環境需求
確保你的環境安裝了以下依賴項：

你可以使用 `requirements.txt` 來安裝所有依賴：
```bash
pip install -r requirements.txt
```

## 🚀 安裝與使用方法


## 📌 TODO 事項


## 📝 版權與許可
本專案僅供學術研究與個人學習使用，請勿用於商業用途。

