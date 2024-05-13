import pandas as pd
import json

data = []

with open('105_first.json', 'r', encoding='utf-8') as file:
    for line in file:
        try:
            # Преобразование каждой строки в JSON объект
            json_obj = json.loads(line)
            if json_obj['text']:
                data.append([json_obj['text'], '[]'])
        except json.JSONDecodeError:
            # Обработка случаев, когда строка не является допустимым JSON объектом
            continue

# Создание DataFrame из списка JSON объектов
df = pd.DataFrame(data)
df.to_csv('preprocessed_data.csv', header=False, index=False)
