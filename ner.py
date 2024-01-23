import fastai
from fastai import *
from fastai.text import *
import pandas as pd
import numpy as np
from functools import partial
import io
import os
import json

data = []

with open('105_first.json', 'r', encoding='utf-8') as file:
    for line in file:
        try:
            # Преобразование каждой строки в JSON объект
            json_obj = json.loads(line)
            data.append(json_obj)
        except json.JSONDecodeError:
            # Обработка случаев, когда строка не является допустимым JSON объектом
            continue

# Создание DataFrame из списка JSON объектов
df = pd.DataFrame(data)

# Вывод первых нескольких строк DataFrame для проверки
# print(dataframe.head())
