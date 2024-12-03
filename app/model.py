import os
from datasets import load_dataset

def load_and_preprocess_data():
    dataset = load_dataset("conll2003")
    sentences, entity_sequences = [], []

    for split in ["train", "validation", "test"]:
        for record in dataset[split]:
            tokens = record["tokens"]
            ner_tags = record["ner_tags"]
            tags = [dataset["train"].features["ner_tags"].feature.int2str(tag) for tag in ner_tags]

            sentence = " ".join(tokens)
            entity_seq = [f"{token} <sep> {tag}" for token, tag in zip(tokens, tags) if tag != "O"]
            sentences.append(sentence)
            entity_sequences.append(" <sep> ".join(entity_seq))

    return sentences, entity_sequences

def split_data(sentences, entity_sequences, test_size=0.2):
    from sklearn.model_selection import train_test_split
    return train_test_split(sentences, entity_sequences, test_size=test_size, random_state=42)
