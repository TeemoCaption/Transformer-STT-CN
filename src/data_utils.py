from datasets import load_dataset, Audio, concatenate_datasets
from tqdm.auto import tqdm
from transformers import Wav2Vec2CTCTokenizer
import json
import re  # 引入 re 模組以使用正則表達式

def load_commonvoice_datasets():
    cv_zh_train = load_dataset("mozilla-foundation/common_voice_16_1", "zh-TW", split="train", trust_remote_code=True)
    cv_zh_valid = load_dataset("mozilla-foundation/common_voice_16_1", "zh-TW", split="validation", trust_remote_code=True)
    cv_zh_test = load_dataset("mozilla-foundation/common_voice_16_1", "zh-TW", split="test", trust_remote_code=True)

    cv_tai_train = load_dataset("mozilla-foundation/common_voice_16_1", "nan-tw", split="train", trust_remote_code=True)
    cv_tai_valid = load_dataset("mozilla-foundation/common_voice_16_1", "nan-tw", split="validation", trust_remote_code=True)
    cv_tai_test = load_dataset("mozilla-foundation/common_voice_16_1", "nan-tw", split="test", trust_remote_code=True)

    return (cv_zh_train, cv_zh_valid, cv_zh_test, cv_tai_train, cv_tai_valid, cv_tai_test)

def merge_datasets(cv_zh, cv_tai, split_name="train"):
    print(f"合併 {split_name} 資料集...")
    for _ in tqdm([1, 2], desc=f"合併 {split_name}"):
        pass
    merged_dataset = concatenate_datasets([cv_zh, cv_tai])
    return merged_dataset

def clean_sentence(example):
    example['sentence'] = re.sub(r'（.*?）', '', example['sentence']).strip()
    return example

def preprocess_dataset(train_dataset, valid_dataset, test_dataset):
    train_dataset = train_dataset.map(clean_sentence)
    valid_dataset = valid_dataset.map(clean_sentence)
    test_dataset = test_dataset.map(clean_sentence)

    keep_cols = ["audio", "sentence"]
    cols_to_remove = [col for col in train_dataset.column_names if col not in keep_cols]

    train_dataset = train_dataset.remove_columns(cols_to_remove)
    valid_dataset = valid_dataset.remove_columns(cols_to_remove)
    test_dataset = test_dataset.remove_columns(cols_to_remove)

    train_dataset = train_dataset.cast_column("audio", Audio(sampling_rate=16000))
    valid_dataset = valid_dataset.cast_column("audio", Audio(sampling_rate=16000))
    test_dataset = test_dataset.cast_column("audio", Audio(sampling_rate=16000))

    return train_dataset, valid_dataset, test_dataset

def build_vocab(train_dataset):
    all_texts = " ".join(train_dataset["sentence"])
    vocab_chars = sorted(set(all_texts))
    return vocab_chars

def create_and_save_vocab(train_dataset, vocab_json_path="vocab.json"):
    vocab_chars = build_vocab(train_dataset)
    print(f"字元總數: {len(vocab_chars)}")

    if " " in vocab_chars:
        vocab_chars.remove(" ")
        if "|" not in vocab_chars:
            vocab_chars.append("|")
        print("將空格替換為 '|'")

    vocab_chars = sorted(vocab_chars)
    vocab_dict = {"[PAD]": 0}
    for idx, char in enumerate(vocab_chars, start=1):
        vocab_dict[char] = idx
    vocab_dict["[UNK]"] = len(vocab_dict)

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

def filter_invalid_chars(dataset, processor):
    """
    自動清除不在詞彙表中的字元，避免訓練時出現 vocab 超出錯誤
    """
    vocab = processor.tokenizer.get_vocab()
    valid_chars = set(vocab.keys())

    def clean_invalid_chars(example):
        original_sentence = example["sentence"]
        cleaned_sentence = ''.join([char for char in original_sentence if char in valid_chars])
        if cleaned_sentence.strip() == "":
            cleaned_sentence = "[PAD]"
        example["sentence"] = cleaned_sentence
        return example

    print("開始清除所有不在詞彙表中的字元...")
    dataset = dataset.map(clean_invalid_chars, desc="清除不合法字元") 
    print("清除完成")
    return dataset

