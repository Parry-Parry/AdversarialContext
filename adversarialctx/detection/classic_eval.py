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
from nltk import sent_tokenize

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

### END CONVIENIENCE FUNCTIONS ###


def main(modelpath, 
         datapath : str, 
         out : str, 
         modeltype : str, 
         type : str, 
         dataset : str = None, 
         context : bool = False, 
         window_size : int = 0, 
         max : bool = True, 
         sentence = False, 
         injection_type : str = "", 
         nature : str = ""):
    global device
    ### BEGIN LOOKUPS AND MODELS INIT ###
    ds = ir_datasets.load(dataset)
    docs = pd.DataFrame(ds.docs_iter()).set_index('doc_id').text.to_dict()

    cols = ['qid', 'docno', 'adversary', 'rel', 'pos', 'salience', 'salience_type', 'sentence', 'context']
    if context: cols = ['qid', 'docno', 'adversary', 'sentence', 'rel', 'pos', 'salience', 'salience_type', 'context']
    with open(datapath, 'r') as f:
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
    
    score_func = score_func if window_size == 0 and not sentence else init_slide(modeltype, window_size, max, sentence)
    
    ### END LOOKUPS AND MODELS INIT ###
    
    sets = []
    if type == 'salience':
        for sal in ['salient', 'nonsalient']:
            tmp = texts[texts.salience==sal]
            for pos in ['before', 'after']:
                tmptmp = tmp[tmp.pos==pos]
                sets.append(tmptmp.copy())
    else:
        for pos in ['before', 'middle', 'after']:
            tmp = texts[texts.pos==pos]
            sets.append(tmp.copy())
    for subsubset in sets:
        frames = []
        for ctx in texts.context.unique().tolist():
            subsubsubset = subsubset[subsubset.context==ctx]
            position = subsubsubset.pos.tolist()[0]
            salience = subsubsubset.salience.tolist()[0]
            
        pd.concat(frames).to_csv(os.path.join(out, f'{nature}.{injection_type}.{modeltype}.{position}.{salience}.csv'))

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
    fire.Fire(main)