import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer, get_scheduler
from datasets import Dataset
import torch
from torch import nn
import evaluate

def to_categorical(row, n_class):
    row['label'] = np.eye(n_class)[row['label']]
    return row

def prepare_data(data, n_class, device, eval_size=0.1, test=False):
    from datasets import ClassLabel
    records = {'text':[], 'label':[]}
    X, Y = data
    for x, y in zip(X, Y):
        records['text'].append(x)
        records['label'].append(y)
    ds = Dataset.from_dict(records).shuffle(seed=42)
    if test: 
        #ds = ds.map(lambda x : to_categorical(x, n_class), batch_size=1)
        return ds.with_format("torch", device=device)
    else:
        features = ds.features.copy()
        features['label'] = ClassLabel(num_classes=n_class)
        ds = ds.cast(features)
        splits = ds.train_test_split(test_size=eval_size, stratify_by_column="label")
        for k, d in splits.items():
            #tmp = d.map(lambda x : to_categorical(x, n_class), batch_size=1)
            splits[k] = d.with_format("torch", device=device)
        return splits

def tokenize(set, tokenizer):
    return tokenizer(set['text'], truncation=True)  

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    metrics = evaluate.combine(["accuracy", "recall", "precision", "f1"])

    return metrics.compute(predictions=predictions, references=labels)
    
def train_bert(data, **kwargs):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    name = kwargs.pop('model_id')
    epochs = kwargs.pop('epochs', 1)
    lr = kwargs.pop('lr', 1e-5)
    n_class = kwargs.pop('n_class', 2)

    ds = prepare_data(data, n_class, device)
    train = ds['train']
    eval = ds['test']
    model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=n_class).to(device)
    tokenizer = AutoTokenizer.from_pretrained(name)
    enc_train = train.map(lambda x : tokenize(x, tokenizer), batched=True)
    enc_eval = eval.map(lambda x : tokenize(x, tokenizer), batched=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    lr_schedule = get_scheduler('linear', optimizer=optimizer, num_training_steps=len(train)*epochs, num_warmup_steps=int(len(train)//4))
    training_args = TrainingArguments(output_dir=kwargs.pop('out_dir'), 
                                      per_device_train_batch_size=kwargs.pop('batch_size', 8), 
                                      evaluation_strategy='epoch',
                                      save_strategy = 'epoch', 
                                      num_train_epochs=epochs,
                                      load_best_model_at_end=True)

    trainer = Trainer(model, 
                      args=training_args, 
                      train_dataset=enc_train, 
                      eval_dataset=enc_eval, 
                      tokenizer=tokenizer, 
                      optimizers=(optimizer, lr_schedule),
                      compute_metrics=compute_metrics)
    trainer.train()

    return [model]

def test_bert(data, model, **kwargs):
    from evaluate import evaluator
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    test = prepare_data(data, kwargs.pop('n_class', 2), device, test=True)
    tokenizer = AutoTokenizer.from_pretrained(kwargs.pop('model_id'))
    enc_test = test.map(lambda x : tokenize(x, tokenizer), batched=True)
    task_evaluator = evaluator("text-classification")
    return task_evaluator.compute(model, enc_test, evaluate.combine(["accuracy", "recall", "precision", "f1"]), tokenizer=tokenizer)


