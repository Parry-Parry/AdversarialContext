from functools import partial
import pyterrier as pt
pt.init()
import pickle
import fire 
import os
import re
from typing import NamedTuple

import pandas as pd
import torch 
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import ir_datasets

from util import *

### EVAL MODELS ###

class cfg(NamedTuple):
    name : str
    dataset : str
    checkpoint : str 
    gpu : bool

def clean_text(text):
    text = re.sub(r'[^A-Za-z0-9 ]+', '', text)
    return re.sub(r'/[^\x00-\x7F]/g', '', text).strip()

def init_colbert(config):
    from pyterrier_colbert.ranking import ColBERTModelOnlyFactory
    pytcolbert = ColBERTModelOnlyFactory(config.checkpoint, gpu=config.gpu)
    preprocess = lambda x : x
    return pytcolbert.text_scorer(), preprocess

def init_dr(config):
    from pyterrier_dr import ElectraScorer, TasB
    if config.name == 'electra': 
        scorer = ElectraScorer()
    else: 
        assert config.checkpoint is not None
        scorer = TasB(config.checkpoint) 

    preprocess = lambda x : x
    return scorer, preprocess
    
def init_monot5(config):
    from pyterrier_t5 import MonoT5ReRanker
    preprocess = lambda x : x
    return MonoT5ReRanker(), preprocess

def init_bm25(config):
    ds = pt.get_dataset(config.dataset)
    indx = pt.IndexFactory.of(ds.get_index(variant='terrier_stemmed'))
    scorer = pt.batchretrieve.TextScorer(body_attr='text', wmodel='BM25', background_index=indx, properties={"termpipelines" : "Stopwords,PorterStemmer"})
    return scorer, clean_text

scorers = {
    'tasb' : init_dr,
    'electra' : init_dr,
    't5' : init_monot5,
    'colbert' : init_colbert,
    'bm25' : init_bm25
}

### END EVAL MODELS ###

def main(retriever : str,
         candidatepath : str, 
         originalpath : str, 
         outputpath : str, 
         modeltype : str, 
         modelpath : str, 
         context : bool = False, 
         window_size : int = 0, 
         sentence : bool = False, 
         maximum : bool = False,
         dataset : str = 'msmarco-passage',
         dataset_lookup : str = 'msmarco_passage',
         alpha : float = 0.2,
         checkpoint : str = None,
         gpu : bool = None):
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    config = cfg(retriever, dataset_lookup, checkpoint, gpu=torch.cuda.is_available() if not gpu else gpu)
    scorer, preprocess = scorers[retriever](config)

    ds = ir_datasets.load(dataset)
    queries = {q.qid : q.query for q in ds.queries_iter()}

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

    score_func = score_func if window_size == 0 and not sentence else init_slide(modeltype, window_size, maximum, sentence)
    
    propa_score = partial(score_func, model=model, encoder=encoder)
    scorer = build_generic_apply_pipe(scorer, propa_score, partial(priorityfusion, alpha=alpha))

    with open(originalpath, 'r') as f:
        items = map(lambda x : x.rstrip().split('\t'), f.readlines())
    maxd = max(items , key=lambda x : int(x[2])) + 1

    with open(candidatepath, 'r') as f:
        items = map(lambda x : x.rstrip().split('\t'), f.readlines())
    cols = ['qid', 'docno', 'adversary', 'rel', 'pos', 'salience', 'salience_type', 'sentence', 'context']
    if context: cols = ['qid', 'docno', 'adversary', 'sentence', 'rel', 'pos', 'salience', 'salience_type', 'context']
    vals = list(map(list, zip(*items)))
    texts = pd.DataFrame.from_dict({r : v for r, v in zip(cols, vals)})

    unique_docs = {docno : maxd+i for i, docno in enumerate(texts['docno'].unique().tolist())}
    texts['docno'] = texts['docno'].apply(lambda x : unique_docs[x])

    data = pd.DataFrame.from_records(list(map(lambda x : {'qid': x.qid, 'query' : queries[x.qid], 'docno' : x.docno, 'text' : x.adversary, 'context': x.context, 'pos' : x.pos, 'salience' : x.salience}), texts.itertuples()))
    data['query'] = data['query'].apply(preprocess)
    data['text'] = data['text'].apply(preprocess)
    scores = scorer.transform(data)

    with open(outputpath, 'w') as f:
        for score in scores.itertuples():
            f.write(f'{score.qid}\t{score.docno}\t{score.score}\t{score.context}\t{score.pos}\t{score.salience}\n')

if __name__ == "__main__":
    fire.Fire(main)
