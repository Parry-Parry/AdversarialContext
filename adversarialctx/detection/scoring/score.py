import pickle
import fire 
import os

import pandas as pd
import torch 
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import ir_datasets
import numpy as np
from scipy.special import softmax
from nltk import word_tokenize, sent_tokenize

from dataclasses import dataclass

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

def main(modelpath : str, 
         modeltype : str, 
         datasetpath : str, 
         outpath : str, 
         dataset : str = 'msmarco-passage',
         standard : bool = False,
         context : bool = False, 
         window_size : int = 0, 
         max : bool = True, 
         sentence : bool = False):
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    if modeltype == 'bert':
        model = AutoModelForSequenceClassification.from_pretrained(modelpath)
        model.to(device)
        encoder = AutoTokenizer.from_pretrained(modelpath)
        score_func = score_bert 
    else: 
        device = None
        with open(os.path.join(modelpath, 'model.pkl'), 'rb') as f:
            model = pickle.load(f)
        with open(os.path.join(modelpath, 'encoder.pkl'), 'rb') as f:
            encoder = pickle.load(f)
        score_func = score_regression 

    ds = ir_datasets.load(dataset)
    docs = pd.DataFrame(ds.docs_iter()).set_index('doc_id').text.to_dict()

    if modeltype == 'bert': score_func = score_bert
    else: score_func = score_regression

    with open(datasetpath, 'r') as f:
        items = map(lambda x : x.rstrip().split('\t'), f.readlines())
        if standard:
            data = list(map(lambda x : Item(x[0], x[1], docs[x[2]]), items))
        else:
            cols = ['qid', 'docno', 'adversary', 'rel', 'pos', 'salience', 'salience_type', 'sentence', 'context']
            if context: cols = ['qid', 'docno', 'adversary', 'sentence', 'rel', 'pos', 'salience', 'salience_type', 'context']
            vals = list(map(list, zip(*items)))
            texts = pd.DataFrame.from_dict({r : v for r, v in zip(cols, vals)})
            data = list(map(lambda x : Item(x.qid, x.docno, x.adversary), texts.itertuples()))
        
    score_func = score_func if window_size == 0 and not sentence else init_slide(modeltype, window_size, max, sentence)

    out = []
    for item in data: # item composed of qid, docno and text
        score = score_func(model, encoder, item.text)
        out.append((item.qid, item.docno, score))

    with open(outpath, 'w') as f:
        for item in out: f.write(f'{item[0]}\t{item[1]}\t{item[2]}\n')
    

if __name__ == '__main__':
    fire.Fire(main) 