from datasets import load_dataset, Audio, concatenate_datasets
from tqdm.auto import tqdm
from transformers import Wav2Vec2CTCTokenizer
import json


def load_commonvoice_datasets():
    """
    載入 Common Voice 16.1 中文(臺灣) 與 臺語(閩南語) 的資料集，
    並分別返回訓練、驗證、測試集。
    """
    # 載入中文(臺灣)資料集
    cv_zh_train = load_dataset(
        "mozilla-foundation/common_voice_16_1",
        "zh-TW",
        split="train",
        trust_remote_code=True
    )
    cv_zh_valid = load_dataset(
        "mozilla-foundation/common_voice_16_1",
        "zh-TW",
        split="validation",
        trust_remote_code=True
    )
    cv_zh_test = load_dataset(
        "mozilla-foundation/common_voice_16_1",
        "zh-TW",
        split="test",
        trust_remote_code=True
    )

    # 載入臺語(閩南語)資料集
    cv_tai_train = load_dataset(
        "mozilla-foundation/common_voice_16_1",
        "nan-tw",
        split="train",
        trust_remote_code=True
    )
    cv_tai_valid = load_dataset(
        "mozilla-foundation/common_voice_16_1",
        "nan-tw",
        split="validation",
        trust_remote_code=True
    )
    cv_tai_test = load_dataset(
        "mozilla-foundation/common_voice_16_1",
        "nan-tw",
        split="test",
        trust_remote_code=True
    )
    
    return (cv_zh_train, cv_zh_valid, cv_zh_test,
            cv_tai_train, cv_tai_valid, cv_tai_test)


def merge_datasets(cv_zh, cv_tai, split_name="train"):
    """
    合併中文與臺語資料集，並回傳合併後的資料集
    """
    print(f"合併 {split_name} 資料集...")
    for _ in tqdm([1, 2], desc=f"合併 {split_name}"):
        pass
    merged_dataset = concatenate_datasets([cv_zh, cv_tai])
    return merged_dataset


def preprocess_dataset(train_dataset, valid_dataset, test_dataset):
    """
    移除不必要欄位，只保留 audio 與 sentence，
    並將 audio 轉換成 16kHz 的取樣率
    """
    # 保留的欄位
    keep_cols = ["audio", "sentence"]
    # 找出不需要的欄位（假設三個資料集欄位相同）
    cols_to_remove = [col for col in train_dataset.column_names if col not in keep_cols]
    
    train_dataset = train_dataset.remove_columns(cols_to_remove)
    valid_dataset = valid_dataset.remove_columns(cols_to_remove)
    test_dataset  = test_dataset.remove_columns(cols_to_remove)
    
    # cast audio 為 16kHz
    train_dataset = train_dataset.cast_column("audio", Audio(sampling_rate=16000))
    valid_dataset = valid_dataset.cast_column("audio", Audio(sampling_rate=16000))
    test_dataset  = test_dataset.cast_column("audio", Audio(sampling_rate=16000))
    
    return train_dataset, valid_dataset, test_dataset


def build_vocab(train_dataset):
    """
    從訓練集中的 sentence 欄位建立字元集合
    """
    all_texts = " ".join(train_dataset["sentence"])
    vocab_chars = sorted(set(all_texts))
    return vocab_chars

def create_and_save_vocab(train_dataset, vocab_json_path="vocab.json"):
    """
    根據訓練資料建立字元集合、詞彙表字典，
    並存成 JSON 檔，同時建立 Wav2Vec2CTCTokenizer。
    """
    # 從訓練資料建立字元集合
    vocab_chars = build_vocab(train_dataset)
    print(f"字元總數: {len(vocab_chars)}")
    
    # 建立 {字元: 索引} 詞彙表字典
    vocab_dict = {char: idx for idx, char in enumerate(vocab_chars)}
    
    # 將空格替換為可見符號，例如用 '|' 表示空格
    if " " in vocab_dict:
        space_index = vocab_dict[" "]
        vocab_dict["|"] = space_index
        del vocab_dict[" "]
        print(f"將空格替換為 '|'，索引為 {space_index}")
    
    # 添加 CTC Blank (使用 [PAD] 作為 blank) 和 [UNK] 符號
    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)
    print(f"最終詞彙表大小: {len(vocab_dict)}")
    
    # 將詞彙表字典存為 JSON 檔
    with open(vocab_json_path, "w", encoding="utf-8") as f:
        json.dump(vocab_dict, f, ensure_ascii=False)
    
    # 使用詞彙表建立 Wav2Vec2CTCTokenizer
    tokenizer = Wav2Vec2CTCTokenizer(vocab_json_path, 
                                     unk_token="[UNK]", 
                                     pad_token="[PAD]", 
                                     word_delimiter_token="|")
    return tokenizer, vocab_dict