import pandas as pd
import numpy as np
import transformers 
import logging
from transformers.trainer import logger as noisy_logger

import params
from prepare_data import TokenLabelDataset
from compute_metrics import compute_metrics


model = transformers.AutoModelForTokenClassification.from_pretrained(params.MODEL_CKECKPOINT, num_labels=len(params.LABEL_LIST))
model.config.id2label = dict(enumerate(params.MODEL_CKECKPOINT))
model.config.label2id = {v: k for k, v in model.config.id2label.items()}

tokenizer = transformers.AutoTokenizer.from_pretrained(params.MODEL_CKECKPOINT)

''' Training '''
num_epochs = 3

args = transformers.TrainingArguments(
    "ner",
    evaluation_strategy = "epoch",
    output_dir=params.CHECKPOINTS_DIR,
    learning_rate = params.LR,
    per_device_train_batch_size = params.BATCH_SIZE,
    per_device_eval_batch_size = params.BATCH_SIZE,
    num_train_epochs = num_epochs, # epoch count !!!
    weight_decay = params.WEIGHT_DECAY,
)

data_collator = transformers.DataCollatorForTokenClassification(tokenizer)

trainer = transformers.Trainer(
    model,
    args,
    train_dataset = TokenLabelDataset(params.SAVE_DIR + 'tokenized_train.pkl'),
    eval_dataset = TokenLabelDataset(params.SAVE_DIR + 'tokenized_validation.pkl'),
    data_collator = data_collator,
    tokenizer = tokenizer,
    compute_metrics = compute_metrics
)

print(f"Training for {num_epochs} epochs:")

noisy_logger.setLevel(logging.WARNING)
trainer.train()

''' Evaluating '''
print("Test evaluation all:")
print(trainer.evaluate())

print("Test evaluation per category:")
predictions, labels, _ = trainer.predict(pd.read_pickle(params.SAVE_DIR + 'tokenized_test.pkl'))
predictions = np.argmax(predictions, axis=2)

# Remove ignored index (special tokens)
true_predictions = [
    [params.LABEL_LIST[p] for (p, l) in zip(prediction, label) if l != -100]
    for prediction, label in zip(predictions, labels)
]
true_labels = [
    [params.LABEL_LIST[l] for (p, l) in zip(prediction, label) if l != -100]
    for prediction, label in zip(predictions, labels)
]

results = params.METRIC.compute(predictions=true_predictions, references=true_labels)

print(results)