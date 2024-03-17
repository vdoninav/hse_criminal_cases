import pandas as pd
import json
import ast
from tqdm import tqdm

import prepare_data
import params

if params.RENEW_SAVED_DATA_IN_PREPROCESSING:
    """ Lading datasets """
    RuLegalNER_train = pd.read_csv(params.TRAIN_PATH, header=None, names=["text", "entities"])
    RuLegalNER_test = pd.read_csv(params.TEST_PATH, header=None,  names=["text", "entities"])
    RuLegalNER_validation = pd.read_csv(params.VALIDATION_PATH, header=None,  names=["text", "entities"])

    RuLegalNER_train['entities'] = RuLegalNER_train['entities'].apply(ast.literal_eval)
    RuLegalNER_test['entities'] = RuLegalNER_test['entities'].apply(ast.literal_eval)
    RuLegalNER_validation['entities'] = RuLegalNER_validation['entities'].apply(ast.literal_eval)

    RuLegalNER_train['entities'] = RuLegalNER_train['entities'].apply(lambda entities: [dict(zip(params.KEYS, values + [params.TYPES[values[3]]])) for values in entities])
    RuLegalNER_test['entities'] = RuLegalNER_test['entities'].apply(lambda entities: [dict(zip(params.KEYS, values + [params.TYPES[values[3]]])) for values in entities])
    RuLegalNER_validation['entities'] = RuLegalNER_validation['entities'].apply(lambda entities: [dict(zip(params.KEYS, values + [params.TYPES[values[3]]])) for values in entities])


    """ Tokenizing datasets + saving them """
    ner_train = pd.DataFrame([prepare_data.extract_labels(RuLegalNER_train.loc[i]) for i in tqdm(range(RuLegalNER_train.shape[0]), desc='Extracting train labels')])
    ner_train.to_pickle(params.SAVE_DIR + "tokenized_train.pkl")
    del RuLegalNER_train
    print('saved')

    ner_test = pd.DataFrame([prepare_data.extract_labels(RuLegalNER_test.loc[i]) for i in tqdm(range(RuLegalNER_test.shape[0]), desc='Extracting test labels')])
    ner_test.to_pickle(params.SAVE_DIR + "tokenized_test.pkl")
    del RuLegalNER_test
    print('saved')

    ner_validation = pd.DataFrame([prepare_data.extract_labels(RuLegalNER_validation.loc[i]) for i in tqdm(range(RuLegalNER_validation.shape[0]), desc='Extracting valid labels')])
    ner_validation.to_pickle(params.SAVE_DIR + "tokenized_validation.pkl")
    del RuLegalNER_validation
    print('saved')

    example = ner_validation.loc[5:7]
    example.to_pickle(params.SAVE_DIR + "tokenized_example.pkl")


