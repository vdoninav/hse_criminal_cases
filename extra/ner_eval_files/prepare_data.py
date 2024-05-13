from torch.utils.data import Dataset
import pandas as pd
from razdel import tokenize

import params

def extract_labels(item):
    raw_toks = list(tokenize(item.text))
    words = [tok.text for tok in raw_toks]
    word_labels = ['O'] * len(raw_toks)
    char2word = [None] * len(item.text)
    for i, word in enumerate(raw_toks):
        char2word[word.start:word.stop] = [i] * len(word.text)

    for e in item.entities:
        e_words = sorted({idx for idx in char2word[e['start']:e['end']] if idx is not None})
        word_labels[e_words[0]] = 'B-' + e['entity_type'] # begining
        for idx in e_words[1:]:
            word_labels[idx] = 'I-' + e['entity_type'] # internal

    return {'tokens': words, 'labels': word_labels}

def tokenize_and_align_labels(example, tokenizer, label_all_tokens=True):
    tokenized_inputs = tokenizer(example["tokens"], truncation=True, is_split_into_words=True)

    word_ids = tokenized_inputs.word_ids()
    previous_word_idx = None
    label_ids = []

    for word_idx in word_ids:
        # Special tokens have a word id that is None. We set the label to -100 so they are automatically
        # ignored in the loss function.
        if word_idx is None:
            label_ids.append(-100)
        # We set the label for the first token of each word.
        elif word_idx != previous_word_idx:
            label_ids.append(example['labels'][word_idx])
        # For the other tokens in a word, we set the label to either the current label or -100, depending on
        # the label_all_tokens flag.
        else:
            label_ids.append(example.labels[word_idx] if label_all_tokens else -100)
        previous_word_idx = word_idx

        label_ids = [params.LABEL_LIST.index(idx) if isinstance(idx, str) else idx for idx in label_ids]


    tokenized_inputs["labels"] = label_ids
    return tokenized_inputs

class TokenLabelDataset(Dataset):
    def __init__(self, data_path, tokenizer):
        data = pd.read_pickle(data_path)
        self.data = data
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        item = tokenize_and_align_labels(self.data.loc[idx], self.tokenizer)
        return item

    def __len__(self):
        return len(self.data)

