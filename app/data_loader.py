import os
import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path, sep="\t", header=None, names=["word", "tag"])
    sentences, tags = [], []
    sentence, tag_seq = [], []

    for _, row in data.iterrows():
        if pd.isna(row["word"]):  # New sentence separator
            if sentence:
                sentences.append(" ".join(sentence))
                tags.append(" ".join(tag_seq))
                sentence, tag_seq = [], []
        else:
            sentence.append(row["word"])
            tag_seq.append(row["tag"])

    return sentences, tags

def split_data(sentences, tags, test_size=0.2):
    return train_test_split(sentences, tags, test_size=test_size, random_state=42)
