from scipy.special import softmax
import torch
import numpy as np
from nltk import word_tokenize, sent_tokenize

from dataclasses import dataclass

def interp_fusion(rel : float, prop : float, alpha : float = 0.9, pi : int = 60, linear : bool = False) -> float:
    rel = (rel + 1) / 2  # normalise cosine similarity with theoretical max & min
    if linear: return alpha * rel + (1 - alpha) * prop # Perform linear fusion
    else: return 1 / (pi + alpha * rel) + 1 / (pi + (1 - alpha) * prop) # Perform reciprocal rank fusion

def rrfusion(rel : float, prop : float, alpha : float = 0.9, relpi : int = 4, proppi : int = 10) -> float:
    rel = (rel + 1) / 2  # normalise cosine similarity with theoretical max & min
    return 1 / (relpi + alpha * rel) + 1 / (proppi + (1 - alpha) * prop) # Perform reciprocal rank fusion

def alphafusion(rel : float, prop : float, alpha : float = 0.9):
    return (rel + 1) / 2 + alpha * prop

def priorityfusion(rel : float, prop : float, alpha : float = 0.9):
    pass 

@dataclass
class Item:
    qid: str
    docno: str
    text: str

def score_regression(model, encoder, text):
    x = encoder.transform([text])
    res = model.predict_proba(x)
    return res[0][-1]

def score_bert(model, tokenizer, text):
    global device
    toks = tokenizer(text, return_tensors='pt', truncation=True).to(device)
    with torch.no_grad():
        pred = softmax(torch.flatten(model(**toks).logits).cpu().detach().numpy())
    return pred[1]

def window(seq, window_size=5):
    split_seq = word_tokenize(seq)
    # Code adapted from https://stackoverflow.com/questions/8408117/generate-a-list-of-strings-with-a-sliding-window-using-itertools-yield-and-ite Eumiro 7/12/2011
    for i in range(len(split_seq) - window_size + 1):
        yield ' '.join(split_seq[i:i+window_size])

def sentence_window(seq):
    splits = sent_tokenize(seq)
    for s in splits:
        yield s

def init_slide(model, window_size=5, max=True, sentence=False):
    if model == 'bert': score_func = score_bert
    else: score_func = score_regression
        
    def inner_func(model, encoder, text):
        slide = window(text, window_size) if not sentence else sentence_window(text)
        vals = [score_func(model, encoder, s) for s in slide]
        if len(vals) < 2: return score_func(model, encoder, text)
        if max: return np.amax(vals)
        else: return np.mean(vals)
    return inner_func

def build_generic_apply(model, score_func, fusion_func):
    pass