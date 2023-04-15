import pickle
import fire 
import os

import pandas as pd
import torch 
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import ir_datasets

from util import *

def main(candidatepath : str, 
         originalpath : str, 
         outputpath : str, 
         modeltype : str, 
         modelpath : str, 
         context : bool = False, 
         window_size : int = 0, 
         sentence : bool = False, 
         maximum : bool = False,
         dataset : str = 'msmarco-passage'):
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

    with open(originalpath, 'r') as f:
        items = map(lambda x : x.rstrip().split('\t'), f.readlines())

    maxd = max(items , key=lambda x : int(x[2])) + 1
    original = list(map(lambda x : Item(x[0], x[1], docs[x[2]]), items))

    with open(candidatepath, 'r') as f:
        items = map(lambda x : x.rstrip().split('\t'), f.readlines())
    cols = ['qid', 'docno', 'adversary', 'rel', 'pos', 'salience', 'salience_type', 'sentence', 'context']
    if context: cols = ['qid', 'docno', 'adversary', 'sentence', 'rel', 'pos', 'salience', 'salience_type', 'context']
    vals = list(map(list, zip(*items)))
    texts = pd.DataFrame.from_dict({r : v for r, v in zip(cols, vals)})
    new_docnos = [maxd + str(i) for i in range(len(texts))]
    texts['docno'] = new_docnos
    data = list(map(lambda x : Item(x.qid, x.docno, x.adversary), texts.itertuples()))
        
    score_func = score_func if window_size == 0 and not sentence else init_slide(modeltype, window_size, maximum, sentence)

    out = []
    for item in data: # item composed of qid, docno and text
        score = score_func(model, encoder, item.text)
        out.append((item.qid, item.docno, score))
    pass 

if __name__ == "__main__":
    fire.Fire(main)
