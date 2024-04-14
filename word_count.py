from pymystem3 import Mystem

import params
import json


def word_count(word_list):
    out = dict()
    word_list = set(word_list)
    morph = Mystem()

    with open(params.WORD_COUNT_FILE, 'r', encoding='utf-8') as file:
        data = json.load(file)

    for word in word_list:
        word_m = morph.lemmatize(word.lower())[0]
        if word_m in data:
            out[word] = data[word_m]
        else:
            out[word] = 0

    return out
