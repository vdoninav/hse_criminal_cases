from torch.utils.data import Dataset
import pandas as pd
from razdel import tokenize
import pandas as pd
import json
import ast
from tqdm import tqdm

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
    
    return {'words': words, 'labels': word_labels}

def csv_preprocessing(csv_file, save_path):
    """ 
    params: csv_file with 2 columns: "text", "entities" = '[]' 
    """

    """ Lading dataset """
    data = pd.read_csv(csv_file, header=None, names=["text", "entities"])
    data['entities'] = data['entities'].apply(ast.literal_eval)
    data['entities'] = data['entities'].apply(lambda entities: [dict(zip(params.KEYS, values + [params.TYPES[values[3]]])) for values in entities])


    """ Tokenizing datasets + saving it """
    data = pd.DataFrame([extract_labels(data.loc[i]) for i in tqdm(range(data.shape[0]), desc='Extracting labels')])
    print('saved')

    tokens_cnt = 300 # for less truncation
    overlap_cnt = 10

    new_data = pd.DataFrame(columns=data.columns)
    print(data.shape)
    for ind in data.index:
        tokens = data.loc[ind, "words"]
        labels = data.loc[ind, "labels"]
        new_data.loc[len(new_data.index)] = [tokens[:tokens_cnt], labels[:tokens_cnt]]
        tokens = tokens[(tokens_cnt - overlap_cnt):]
        labels = labels[(tokens_cnt - overlap_cnt):]
        while len(tokens) > 0:
            new_data.loc[len(new_data.index)] = [tokens[:tokens_cnt], labels[:tokens_cnt]]
            tokens = tokens[(tokens_cnt - overlap_cnt):]
            labels = labels[(tokens_cnt - overlap_cnt):]
    print(new_data.shape)

    new_data.to_pickle(save_path)

    return save_path


def tokenize_and_align_labels(example, label_all_tokens=True, max_length=512):
    tokenized_inputs = params.TOKENIZER(example["words"], 
                                 is_split_into_words=True,
                                 max_length=max_length,
                                 padding='max_length',
                                 truncation=True)

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
    def __init__(self, data_path):
        data = pd.read_pickle(data_path)
        self.data = data
    def __getitem__(self, idx):
        item = tokenize_and_align_labels(self.data.loc[idx])
        return item

    def __len__(self):
        return len(self.data)

