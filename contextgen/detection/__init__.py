from typing import Any, List
from more_itertools import chunked
import torch 
from nltk.tokenize import sent_tokenize

class Scorer(object):
    def __init__(self, model : Any, tokenizer : Any, window_size : int = 0, batch_size : int = 128, classifier=False) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.window_size = window_size
        self.batch_size = batch_size

        self.score = self.score_window if window_size != 0 else self.score_std

        self.classifier = classifier

    def score_std(self, texts : List[str]) -> List[float]:
        scores = []
        for batch in chunked(texts, self.batch_size):
            inputs = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(self.model.device)
            outputs = self.model(**inputs)
            logits = outputs.logits
            if self.classifier:
                pred = torch.argmax(logits, dim=1)
                scores.extend(pred.tolist())
            else:
                pred = torch.softmax(logits, dim=1)
                scores.extend(pred[:, 1].tolist())
        return scores
    
    def window(self, text : str) -> List[str]:
        return list(chunked(text, self.window_size))
    
    def score_window(self, texts : List[str]) -> float:
        scores = []
        for text in texts:
            if self.window_size == -1: chunks = sent_tokenize(text)
            else: chunks = self.window(text)
            inputs = self.tokenizer(chunks, padding=True, truncation=True, return_tensors="pt").to(self.model.device)
            outputs = self.model(**inputs)
            logits = outputs.logits
            if self.classifier:
                pred = torch.softmax(logits, dim=1)[:, 1]
                scores.append(torch.argmax(pred).item())
            else:
                pred = torch.softmax(logits, dim=1)
                scores.append(max(pred[:, 1].tolist()))
        return scores
    
    def __call__(self, texts : List[str]) -> Any:
        return self.score(texts)