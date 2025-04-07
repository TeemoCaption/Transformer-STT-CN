from datasets import load_dataset, Audio, concatenate_datasets
from tqdm.auto import tqdm
from transformers import Wav2Vec2CTCTokenizer
import json
import re  # 引入 re 模組以使用正則表達式

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


def clean_sentence(example):
    """
    清理標籤中的全形括號及其內容
    """
    example['sentence'] = re.sub(r'（.*?）', '', example['sentence']).strip()
    return example


def preprocess_dataset(train_dataset, valid_dataset, test_dataset):
    """
    清理標籤，移除不必要欄位，只保留 audio 與 sentence，
    並將 audio 轉換成 16kHz 的取樣率
    """
    # 清理標籤
    train_dataset = train_dataset.map(clean_sentence)
    valid_dataset = valid_dataset.map(clean_sentence)
    test_dataset = test_dataset.map(clean_sentence)
    
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
    vocab_chars = build_vocab(train_dataset)
    print(f"字元總數: {len(vocab_chars)}")

    vocab_dict = {char: idx for idx, char in enumerate(vocab_chars)}

    # 替換空格為分隔符號
    if " " in vocab_dict:
        space_index = vocab_dict[" "]
        vocab_dict["|"] = space_index
        del vocab_dict[" "]
        print(f"將空格替換為 '|'，索引為 {space_index}")

    # 加入特殊符號（一定要加進 vocab_dict 裡才能寫入 JSON）
    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)

    print(f"最終詞彙表大小: {len(vocab_dict)}")

    with open(vocab_json_path, "w", encoding="utf-8") as f:
        json.dump(vocab_dict, f, ensure_ascii=False)

    tokenizer = Wav2Vec2CTCTokenizer(
        vocab_json_path,
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token="|"
    )
    return tokenizer, vocab_dict


def debug_check_labels(train_dataset, processor):
    """
    檢查訓練資料中所有句子的字元是否超出 vocab 範圍
    """
    vocab = processor.tokenizer.get_vocab()
    vocab_size = processor.tokenizer.vocab_size
    unk_id = processor.tokenizer.unk_token_id
    errors = []

    print("開始檢查訓練資料中所有句子的字元是否超出 vocab...")
    for i, sample in enumerate(tqdm(train_dataset, desc="檢查中")):
        sentence = sample["sentence"]
        labels = [vocab.get(char, unk_id) for char in sentence]

        for j, token in enumerate(labels):
            if token >= vocab_size:
                print(f"第 {i} 筆資料異常 -> 字元: '{sentence[j]}', token ID: {token}, vocab_size: {vocab_size}")
                errors.append((i, sentence, sentence[j], token))

    print(f"\n檢查完畢，共發現 {len(errors)} 筆異常。")
    return errors