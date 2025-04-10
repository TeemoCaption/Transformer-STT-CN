from datasets import load_dataset, Audio, concatenate_datasets
from tqdm.auto import tqdm
from transformers import Wav2Vec2CTCTokenizer
import json
import re

def load_commonvoice_datasets():
    cv_zh_train = load_dataset("mozilla-foundation/common_voice_16_1", "zh-TW", split="train", trust_remote_code=True)
    cv_zh_valid = load_dataset("mozilla-foundation/common_voice_16_1", "zh-TW", split="validation", trust_remote_code=True)
    cv_zh_test = load_dataset("mozilla-foundation/common_voice_16_1", "zh-TW", split="test", trust_remote_code=True)

    cv_tai_train = load_dataset("mozilla-foundation/common_voice_16_1", "nan-tw", split="train", trust_remote_code=True)
    cv_tai_valid = load_dataset("mozilla-foundation/common_voice_16_1", "nan-tw", split="validation", trust_remote_code=True)
    cv_tai_test = load_dataset("mozilla-foundation/common_voice_16_1", "nan-tw", split="test", trust_remote_code=True)

    return (cv_zh_train, cv_zh_valid, cv_zh_test,
            cv_tai_train, cv_tai_valid, cv_tai_test)

def merge_datasets(cv_zh, cv_tai, split_name="train"):
    print(f"合併 {split_name} 資料集...")
    for _ in tqdm([1, 2], desc=f"合併 {split_name}"):
        pass
    merged_dataset = concatenate_datasets([cv_zh, cv_tai])
    return merged_dataset

def clean_sentence(example):
    # 移除（全形括號）以及其中內容
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

    # 統一轉為 16kHz
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

    # 如果有空格，就轉成 '|'
    if " " in vocab_chars:
        vocab_chars.remove(" ")
        if "|" not in vocab_chars:
            vocab_chars.append("|")
        print("將空格替換為 '|'")

    vocab_chars = sorted(vocab_chars)

    # 先預留 [PAD], [UNK]，然後從第三個位置開始編號
    vocab_dict = {
        "[PAD]": 0,
        "[UNK]": 1
    }
    for idx, char in enumerate(vocab_chars, start=2):
        vocab_dict[char] = idx

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

def debug_check_labels(dataset, processor):
    """
    檢查 dataset 中所有句子的字元是否超出 tokenizer 的 vocab 範圍。
    如有發現，印出字元、token ID、以及 vocab_size。
    """
    vocab = processor.tokenizer.get_vocab()
    vocab_size = processor.tokenizer.vocab_size
    unk_id = processor.tokenizer.unk_token_id
    errors = []

    print("開始檢查資料集中所有句子的字元是否超出 vocab ...")
    for i, sample in enumerate(tqdm(dataset, desc="檢查中")):
        sentence = sample["sentence"]
        labels = [vocab.get(char, unk_id) for char in sentence]
        for j, token in enumerate(labels):
            if token >= vocab_size:
                print(f"第 {i} 筆資料 -> 超出範圍字元: '{sentence[j]}', token ID: {token}, vocab_size: {vocab_size}")
                errors.append((i, sentence, sentence[j], token))

    if errors:
        print(f"檢查完畢，共有 {len(errors)} 筆句子內含超出範圍的字元。")
    else:
        print("檢查完畢，未發現超出範圍的字元。")

    return errors

def filter_invalid_chars(dataset, processor):
    """
    清除不在 vocab 裡的字元（直接過濾掉或替換為 [PAD]/[UNK] 也可）。
    這裡示範「若字元不在 vocab，即移除該字元」。
    如果整句都被清空，就用 [PAD] 代替。
    """
    vocab = processor.tokenizer.get_vocab()
    valid_chars = set(vocab.keys())  # 所有出現在 vocab 裡的 key

    def clean_invalid(example):
        sent = example["sentence"]
        cleaned = [ch for ch in sent if ch in valid_chars]
        if len(cleaned) == 0:
            # 若整個句子都沒字元可用了，就填個 [PAD]
            cleaned = ["[PAD]"]
        example["sentence"] = "".join(cleaned)
        return example

    print("開始清除 dataset 中不在 vocab 的字元 ...")
    dataset = dataset.map(clean_invalid, desc="清除不合法字元")
    print("清除完成。")
    return dataset
