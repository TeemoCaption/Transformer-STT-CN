# 語音轉文字專案 (Speech2Text Project)

這是一個簡單的語音轉文字 (Speech-to-Text, STT) 實作專案，使用 Python 開發，並基於 Transformer 模型來進行語音識別。

## 🚀 專案介紹
本專案的目標是將語音訊號轉換為對應的文字，並透過機器學習技術來提升準確度。

## 📂 資料集
本專案使用 **LJ Speech Dataset** 作為語音訓練資料集。
- 官方下載連結：[LJ Speech Dataset](https://keithito.com/LJ-Speech-Dataset/)

## 📁 專案目錄結構

```
/speech2text_project
│── main.py               # 主程式，負責模型訓練與測試
│── model.py              # Transformer 語音轉文字模型
│── utils.py              # 存放輔助函數 (學習率調整 & 訓練過程顯示)
│── dataset/              # 存放數據集
│── checkpoints/          # 存放模型檢查點
│── requirements.txt      # 依賴項
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

