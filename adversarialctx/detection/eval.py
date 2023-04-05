import pyterrier as pt
pt.init()
from collections import defaultdict
import logging
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import pickle
import os
import ir_datasets
import pandas as pd
import fire
import numpy as np
from scipy.special import softmax
from nltk import word_tokenize
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

### BEGIN CONVIENIENCE FUNCTIONS ###

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

def build_from_df(frame):
    lookup = defaultdict(dict)
    for row in frame.itertuples(): lookup[row.qid][row.docno] = row.adversary
    return lookup

def window(seq, window_size=5):
    split_seq = word_tokenize(seq)
    # Code adapted from https://stackoverflow.com/questions/8408117/generate-a-list-of-strings-with-a-sliding-window-using-itertools-yield-and-ite Eumiro 7/12/2011
    for i in range(len(split_seq) - window_size + 1):
        yield ' '.join(split_seq[i:i+window_size])

def init_slide(model, window_size=5):
    if model == 'bert': score_func = score_bert
    else: score_func = score_regression
        
    def inner_func(model, encoder, text):
        slide = window(text, window_size)
        vals = [score_func(model, encoder, s) for s in slide]
        return np.amax(vals)
    return inner_func

### END CONVIENIENCE FUNCTIONS ###


def main(modelpath, advpath : str, originalpath : str, out : str, modeltype : str, type : str, dataset : str = None, context : bool = False, window_size : int = 0):
    global device
    ### BEGIN LOOKUPS AND MODELS INIT ###
    ds = ir_datasets.load(dataset)
    docs = pd.DataFrame(ds.docs_iter()).set_index('doc_id').text.to_dict()

    lookup = defaultdict(dict)
    with open(originalpath, 'r') as f:
        items = map(lambda x : x.strip().split('\t'), f.readlines())
    
    for item in items: lookup[item[0]][item[1]] = docs[item[1]]

    cols = ['qid', 'docno', 'adversary', 'rel', 'pos', 'salience', 'salience_type', 'sentence', 'context']
    if context: cols = ['qid', 'docno', 'adversary', 'sentence', 'rel', 'pos', 'salience', 'salience_type', 'context']
    with open(advpath, 'r') as f:
        text_items = map(lambda x : [y.strip('\n') for y in x.split('\t')], f.readlines())

    vals = list(map(list, zip(*text_items)))

    texts = pd.DataFrame.from_dict({r : v for r, v in zip(cols, vals)})

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
    
    score_func = score_func if window_size == 0 else init_slide(modeltype, window_size)
    
    ### END LOOKUPS AND MODELS INIT ###
  
    frames = []

    try:
        for ctx in texts.context.unique().tolist():
            subset = texts[texts.context==ctx]
            sets = []
            if type == 'salience':
                for sal in ['salient', 'nonsalient']:
                    tmp = subset[subset.salience==sal]
                    logging.info(f'{sal} : {len(tmp)}')
                    for pos in ['before', 'after']:
                        tmptmp = tmp[tmp.pos==pos]
                        logging.info(f'{pos} : {len(tmptmp)}')
                        sets.append(tmptmp.copy())
            else:
                for pos in ['before', 'middle', 'after']:
                    tmp = subset[subset.pos==pos]
                    sets.append(tmp.copy())
            for subsubsubset in sets:
                adv = build_from_df(subsubsubset)
                position = subsubsubset.pos.tolist()[0]
                salience = subsubsubset.salience.tolist()[0]
                res = []
                for key, item in lookup.items():
                    for doc, text in item.items():
                        original_score = score_func(model, encoder, text)
                        score = score_func(model, encoder, adv[key][doc])
                        res.append({'qid' : key, 'docno' : doc, 'context' : ctx, 'pos' : position, 'salience' : salience, 'orginal_score' : original_score, 'new_score' : score})
                frames.append(pd.DataFrame.from_records(res))
    except ValueError:
        pass
                
    pd.concat(frames).to_csv(out)

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
    fire.Fire(main)