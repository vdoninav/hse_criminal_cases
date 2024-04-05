import pandas as pd
import json
import ast
from tqdm import tqdm

import prepare_data
import params

def test_preprocessing(csv_file, save_path):
    """ params: csv_file with 2 columns: "text", "entities" = '[]' """
    """ Lading datasets """
    data = pd.read_csv(csv_file, header=None, names=["text", "entities"])
    data['entities'] = data['entities'].apply(ast.literal_eval)
    data['entities'] = data['entities'].apply(lambda entities: [dict(zip(params.KEYS, values + [params.TYPES[values[3]]])) for values in entities])


    """ Tokenizing datasets + saving them """
    data = pd.DataFrame([prepare_data.extract_labels(data.loc[i]) for i in tqdm(range(data.shape[0]), desc='Extracting train labels')])
    print('saved')

    tokens_cnt = 510
    overlap_cnt = 10

    new_data = pd.DataFrame(columns=data.columns)
    print(data.shape)
    for ind in data.index:
        tokens = data.loc[ind, "tokens"]
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