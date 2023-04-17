from scipy.special import softmax
import torch
import numpy as np
from nltk import word_tokenize, sent_tokenize

from dataclasses import dataclass

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def interp_fusion(rel : float, prop : float, alpha : float = 0.9, pi : int = 60, linear : bool = False) -> float:
    rel = (rel + 1) / 2  # normalise cosine similarity with theoretical max & min
    if linear: return alpha * rel + (1 - alpha) * prop # Perform linear fusion
    else: return 1 / (pi + alpha * rel) + 1 / (pi + (1 - alpha) * prop) # Perform reciprocal rank fusion

def rrfusion(rel : float, prop : float, alpha : float = 0.9, relpi : int = 4, proppi : int = 10) -> float:
    rel = (rel + 1) / 2  # normalise cosine similarity with theoretical max & min
    return 1 / (relpi + alpha * rel) + 1 / (proppi + (1 - alpha) * prop) # Perform reciprocal rank fusion

def priorityfusion(rel : float, prop : float, alpha : float = 0.9, norm=False):
    if norm: rel = (rel + 1) / 2 
    return rel + alpha * (1 - prop)

@dataclass
class Item:
    qid: str
    docno: str
    text: str

def score_regression(text, model=None, encoder=None):
    x = encoder.transform([text])
    res = model.predict_proba(x)
    return res[0][-1]

def score_bert(text, model=None, tokenizer=None):
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
        
    def inner_func(text, model, encoder):
        slide = window(text, window_size) if not sentence else sentence_window(text)
        vals = [score_func(s, model, encoder) for s in slide]
        if len(vals) < 2: return score_func(text, model, encoder)
        if max: return np.amax(vals)
        else: return np.mean(vals)
    return inner_func


def build_generic_apply_pipe(model, score_func, fusion_func):
    import pyterrier as pt 
    def fuse_scores(df):
        df['score'] = df.apply(lambda x : fusion_func(x['score'], score_func(x['text'])), axis=1)
        return df

    return model >> pt.apply.generic(fuse_scores)

def build_threshold_pipe(model, score_func, t=0.5):
    import pyterrier as pt 
    return model >> pt.apply.generic(lambda x : score_func(x['text']) < t)