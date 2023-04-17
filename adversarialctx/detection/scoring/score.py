import pickle
import fire 
import os
import pandas as pd
import torch 
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import ir_datasets

from util import Item, score_regression, score_bert, init_slide

def main(
         modelpath : str, 
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
            texts = list(map(lambda x : Item(x[0], x[1], docs[x[1]]), items))
        else:
            cols = ['qid', 'docno', 'adversary', 'rel', 'pos', 'salience', 'salience_type', 'sentence', 'context']
            if context: cols = ['qid', 'docno', 'adversary', 'sentence', 'rel', 'pos', 'salience', 'salience_type', 'context']
            vals = list(map(list, zip(*items)))
            texts = pd.DataFrame.from_dict({r : v for r, v in zip(cols, vals)})
        
    score_func = score_func if window_size == 0 and not sentence else init_slide(modeltype, window_size, max, sentence)

    out = []
    if standard:
        for item in texts:
            score = score_func(item.adversary, model, encoder)
            out.append((item.qid, item.docno, score))
        
        with open(outpath, 'w') as f:
            for item in out: f.write(f'{item[0]}\t{item[1]}\t{item[2]}\n')
    else:
        for item in texts.itertuples(): # item composed of qid, docno and text
            score = score_func(item.adversary, model, encoder)
            out.append((item.qid, item.docno, score, item.context, item.pos, item.salience))

        with open(outpath, 'w') as f:
            for item in out: f.write(f'{item[0]}\t{item[1]}\t{item[2]}\t{item[3]}\t{item[4]}\t{item[5]}\n')
    
if __name__ == '__main__':
    fire.Fire(main)