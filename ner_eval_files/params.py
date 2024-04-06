import torch
import evaluate

DEVICE = torch.device('mps' if torch.cuda.is_available() else 'cpu')

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

TRAIN_PATH = '/content/drive/MyDrive/coursework2024/RuLegalNER/train.csv'
TEST_PATH = '/content/drive/MyDrive/coursework2024/RuLegalNER/test.csv'
VALIDATION_PATH = '/content/drive/MyDrive/coursework2024/RuLegalNER/validation.csv'

SAVE_DIR = '/content/drive/MyDrive/coursework2024/data/'

RENEW_SAVED_DATA_IN_PREPROCESSING = False
CHOP_LONG_TEXTS = False

BATCH_SIZE = 100
LR = 2e-5
WEIGHT_DECAY = 0.01

METRIC = evaluate.load("seqeval")

CHECKPOINTS_DIR = '/content/drive/MyDrive/coursework2024/checkpoints/'

# MODEL_CKECKPOINT = "cointegrated/rubert-tiny"

MODEL_CKECKPOINT = 'best-checkpoint-yet'
