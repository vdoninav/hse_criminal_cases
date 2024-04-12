import torch
import transformers

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# IND [2] - Individual
# LE [4] - Legal Entity
# PEN [9] -Penalty
# LAW [13] - Law
# CR [17] - Crime
LABEL_LIST = ['B-CR', 'B-IND', 'B-LAW', 'B-LE', 'B-PEN', 'I-CR', 'I-IND', 'I-LAW', 'I-LE', 'I-PEN', 'O']

KEYS = ["start",
        "end",
        "entity_text",
        "entity_id",
        "entity_type"
        ]
TYPES = {2: "IND",
         4: "LE",
         9: "PEN",
         13: "LAW",
         17: "CR",
         }

# MODEL_CHECKPOINT = "cointegrated/rubert-tiny"
MODEL_CHECKPOINT = 'lebeda/bert-finetuned-on-RuLegalNer'
# MODEL_CHECKPOINT = 'checkpoints/checkpoint-25500'

TOKENIZER = transformers.AutoTokenizer.from_pretrained('cointegrated/rubert-tiny')
