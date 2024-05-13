import pandas as pd
import numpy as np
import transformers
import logging
from transformers.trainer import logger as noisy_logger
import csv

import params
from prepare_data import TokenLabelDataset
from compute_metrics import compute_metrics
from preprocess_data import test_preprocessing

print(params.DEVICE)

model = transformers.AutoModelForTokenClassification.from_pretrained(params.MODEL_CKECKPOINT,
                                                                     num_labels=len(params.LABEL_LIST))
model.config.id2label = dict(enumerate(params.MODEL_CKECKPOINT))
model.config.label2id = {v: k for k, v in model.config.id2label.items()}
model = model.to(params.DEVICE)

tokenizer = transformers.AutoTokenizer.from_pretrained(params.MODEL_CKECKPOINT)

''' Training '''
num_epochs = 0

args = transformers.TrainingArguments(
    evaluation_strategy="epoch",
    output_dir='output/',
    learning_rate=params.LR,
    per_device_train_batch_size=params.BATCH_SIZE,
    per_device_eval_batch_size=params.BATCH_SIZE,
    num_train_epochs=num_epochs,  # epoch count !!!
    weight_decay=params.WEIGHT_DECAY,
)

data_collator = transformers.DataCollatorForTokenClassification(tokenizer)

trainer = transformers.Trainer(
    model,
    args,
    train_dataset=TokenLabelDataset('preprocessed.pkl', tokenizer),
    eval_dataset=TokenLabelDataset('preprocessed.pkl', tokenizer),
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

print(trainer.args.device)

''' Evaluating '''
# print("Test evaluation per category:")
# test_preprocessing('preprocessed_data.csv', 'preprocessed.pkl')
test_dataset = TokenLabelDataset(test_preprocessing('test.csv', 'preprocessed.pkl'), tokenizer)
predictions, labels, _ = trainer.predict(test_dataset)
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

with open('true_predictions.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(true_predictions)

with open('true_labels.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(true_labels)

with open('predictions.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(predictions)

with open('labels.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(labels)

# results = params.METRIC.compute(predictions=true_predictions, references=true_labels)
# print(results)
