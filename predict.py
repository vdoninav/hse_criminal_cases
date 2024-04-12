import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from token_class_pipeline import TokenClassificationPipe
import params


def predict(text_input):
    tokenizer = params.TOKENIZER
    model = AutoModelForTokenClassification.from_pretrained(params.MODEL_CHECKPOINT, num_labels=len(params.LABEL_LIST))
    model.config.id2label = dict(enumerate(params.LABEL_LIST))
    model.config.label2id = {v: k for k, v in model.config.id2label.items()}

    predictions_pipeline = TokenClassificationPipe(model=model, tokenizer=tokenizer, aggregation_strategy='max')

    return predictions_pipeline(text_input)


"""sequence = "Рабочий Жирнов живет один в городе Москве по своим законам. У него есть сводный брат и родная сестра"""

"""[{'entity_group': 'IND',
  'score': 0.5976736,
  'word': 'сводный',
  'start': 72,
  'end': 79},
 {'entity_group': 'IND',
  'score': 0.5692652,
  'word': 'брат',
  'start': 80,
  'end': 84},
 {'entity_group': 'IND',
  'score': 0.52216005,
  'word': 'и',
  'start': 85,
  'end': 86},
 {'entity_group': 'IND',
  'score': 0.7377518,
  'word': 'родная',
  'start': 87,
  'end': 93},
 {'entity_group': 'IND',
  'score': 0.75063056,
  'word': 'сестра',
  'start': 94,
  'end': 100}]"""
