import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from datasets import Dataset
import torch
from torch import nn
import evaluate

metric = evaluate.load("accuracy")

class BertClassifier(nn.Module):
    def __init__(self, model : str, n_class : str, embedding_dim : int = 784, depth : list = [64], **kwargs) -> None:
        super().__init__(**kwargs)
         #model = BertClassifier(, n_class, kwargs.pop('embedding_dim', 784), kwargs.pop('depth'), **kwargs)
        self.encoder = AutoModelForSequenceClassification(model)
        self.classifier = self._make_classifier(embedding_dim, n_class, depth)
    
    def _make_classifier(emb_dim, out_dim, depth, p=0.1):
        layers = []
        in_dim = emb_dim
        for d in depth:
            layers.append(
                nn.Sequential(
                    [
                        nn.Linear(in_dim, d),
                        nn.Dropout(p),
                        nn.ReLu()
                    ]
                )
            )
            in_dim = d
        out_activation = nn.Sigmoid if out_dim == 1 else nn.Softmax
        layers.append(
            nn.Sequential(
                    [
                        nn.Linear(in_dim, out_dim),
                        out_activation()
                    ]
                )
        )
        return nn.Sequential(*layers)
    
    def forward(self, inp):
        x = self.encoder(**inp)
        return self.classifier(x)

def format_dataset(dataset, n_class):
    records = {'text':[], 'label':[]}
    for x, y in dataset:
        records['text'].append(x)
        records['label'].append(np.eye(n_class)[y])
    return Dataset.from_dict(records)
    
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    return metric.compute(predictions=predictions, references=labels)
    
def train_bert(data, **kwargs):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    name = kwargs.pop('model')
    epochs = kwargs.pop('epochs', 1)
    lr = kwargs.pop('lr', 1e-3)
    n_class = kwargs.pop('n_class', 1)

    train, eval = format_dataset(name, data)
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

def test_bert(data, **kwargs):
    pass