import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from datasets import Dataset
import torch
from torch import nn
import evaluate

def format_dataset(data, n_class, device, eval_size=0.1, test=False):
    records = {'text':[], 'label':[]}
    X, Y = data
    for x, y in zip(X, Y):
        records['text'].append(x)
        records['label'].append(np.eye(n_class)[y])
    if test: return Dataset.from_dict(records).with_format("torch", device=device)
    return Dataset.from_dict(records).train_test_split(test_size=eval_size, stratify_by_column="label").with_format("torch", device)
    
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

    ds = format_dataset(name, data, device)
    train = ds['train']
    eval = ds['test']
    model = AutoModelForSequenceClassification(name, num_labels=n_class).to_device(device)
    tokenizer = AutoTokenizer.from_pretrained(name)
    optimizer = torch.optim.AdamW(model, lr=lr)
    training_args = TrainingArguments(output_dir=kwargs.pop('out_dir'), 
                                      per_device_train_batch_size=kwargs.pop('batch_size', 8), 
                                      evaluation_strategy='epochs', 
                                      num_train_epochs=epochs)

    trainer = Trainer(model, 
                      TrainingArguments=training_args, 
                      train_dataset=train, 
                      eval_dataset=eval, 
                      tokenizer=tokenizer, 
                      optimizers=optimizer,
                      compute_metrics=compute_metrics)
    trainer.train()

    return model

def test_bert(data, model, **kwargs):
    from evaluate import evaluator
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    test = format_dataset(data, kwargs.pop('n_class', 2), device, test=True)
    
    task_evaluator = evaluator("text-classification")
    return task_evaluator.compute(model, test, evaluate.combine(["accuracy", "recall", "precision", "f1"]))


