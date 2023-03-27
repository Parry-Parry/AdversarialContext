import pyterrier as pt
pt.init()
from collections import defaultdict
import logging
import torch
import argparse
import os
import ir_datasets
from typing import NamedTuple
import re
import pandas as pd
import numpy as np

### BEGIN EVAL MODELS ###

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

### END EVAL MODELS ###

_logger = ir_datasets.log.easy()

### BEGIN CONVIENIENCE FUNCTIONS ###

def build_data(path):
  result = []
  dataset = ir_datasets.load(path)
  docs = dataset.docs_store()
  queries = {q.query_id: q.text for q in dataset.queries_iter()}
  for qrel in _logger.pbar(ir_datasets.load(path).scoreddocs_iter(), desc='dev data'):
    if qrel.query_id in queries:
      result.append([qrel.query_id, queries[qrel.query_id], qrel.doc_id, docs.get(qrel.doc_id).text])
  return pd.DataFrame(result, columns=['qid', 'query', 'docno', 'text'])

def build_rank_lookup(df):
    frame = {}
    for qid in df.qid.unique().tolist():
        sub = df[df.qid==qid].copy()
        assert len(sub) > 0
        frame[qid] = [(row.docno, row.score) for row in sub.itertuples()]
    return frame

def build_from_df(df, queries):
    new = []
    for row in df.itertuples():
        new.append({'qid':row.qid, 'query':queries[row.qid], 'docno':row.docno, 'text':row.adversary})
    return pd.DataFrame.from_records(new)

def build_from_old(df, queries, documents):
    new = []
    for row in df.itertuples():
        new.append({'qid':row.qid, 'query':queries[row.qid], 'docno':row.docno, 'text':documents[row.docno]})
    return pd.DataFrame.from_records(new)

def get_rank_change(qid, docno, score, lookup):
    old_ranks = [(k, v) for k, v in lookup[qid].items()]
    old_ranks.sort(key=lambda x : x[1], reverse=True)
    old_rank = [i for i, item in enumerate(old_ranks) if item[0]==docno]
    logging.info(f'Found Docno in {len(old_rank)} positions')
    new_ranks = [item for item in old_ranks if item[0] != docno]
    new_ranks.append((docno, score))
    new_ranks.sort(reverse=True, key=lambda x : x[1])
    rank_change = old_rank[0] - [i for i, item in enumerate(new_ranks) if item[0]==docno][0]
    return rank_change
                    
def ABNIRML(score, adv_score):
    diff = score - adv_score
    if diff < 0: return -1 
    elif diff > 0: return 1
    return 0

def get_score(qid, docno, results):
    tmp = results[results['qid']==qid].set_index('docno')['score']
    adv_score = tmp.loc[docno]
    if type(adv_score) != np.float64 and type(adv_score) != np.float32: adv_score = adv_score.values[0]
    return adv_score

scorers = {
    'tasb' : init_dr,
    'electra' : init_dr,
    't5' : init_monot5,
    'colbert' : init_colbert,
    'bm25' : init_bm25
}

### END CONVIENIENCE FUNCTIONS ###

parser = argparse.ArgumentParser()

parser.add_argument('-source', type=str)
parser.add_argument('-full_path', type=str)
parser.add_argument('-scorer', type=str)
parser.add_argument('-type', type=str)
parser.add_argument('-qrels', type=str)
parser.add_argument('-sink', type=str)

parser.add_argument('--dataset', type=str, default=None)
parser.add_argument('--checkpoint', type=str, default=None)
parser.add_argument('--gpu', action='store_true')

def main(args):

    ### BEGIN LOOKUPS AND MODELS INIT ###
    ds = ir_datasets.load(args.qrels)
    queries = pd.DataFrame(ds.queries_iter()).set_index('query_id').text.to_dict()


    lookup_full = defaultdict(dict)
    format_10 = f'{args.scorer}.10.tsv'
    format_50 = f'{args.scorer}.50.tsv'
    with open(os.path.join(args.full_path, format_50), 'r') as f:
        items = map(lambda x : x.split('\t'), f.readlines())
    
    for item in items: lookup_full[item[0]][item[1]] = float(item[2].strip())

    lookup_10 = defaultdict(dict)
    with open(os.path.join(args.full_path, format_10), 'r') as f:
        items = map(lambda x : x.split('\t'), f.readlines())
    
    for item in items: lookup_10[item[0]][item[1]] = float(item[2].strip())

    config = cfg(args.scorer, args.dataset, args.checkpoint, args.gpu)

    try:
        scorer, preprocess = scorers[args.scorer](config)
    except KeyError:
        logging.error(f'Model {args.scorer} not recognised!')
        exit

    ### END LOOKUPS AND MODELS INIT ###

    if 'context' in args.source:
        cols = ['qid', 'docno', 'adversary', 'sentence', 'rel', 'pos', 'salience', 'salience_type', 'context']
    else:
        cols = ['qid', 'docno', 'adversary', 'rel', 'pos', 'salience', 'salience_type', 'sentence', 'context']
    
    with open(args.source, 'r') as f:
        text_items = map(lambda x : [y.strip('\n') for y in x.split('\t')], f.readlines())

    vals = list(map(list, zip(*text_items)))

    texts = pd.DataFrame.from_dict({r : v for r, v in zip(cols, vals)})
    frames = []
    logging.info(f'Pos: {texts.pos.tolist()[:3]}')
    try:
        #texts = pd.read_csv(args.source, sep='\t', header=None, index_col=False, names=cols, dtype=types)
        for ctx in texts.context.unique().tolist():
            subset = texts[texts.context==ctx]
            sets = []
            if args.type == 'salience':
                for sal in ['salient', 'nonsalient']:
                    tmp = subset[subset.salience==sal]
                    for pos in ['before', 'after']:
                        tmptmp = tmp[tmp.pos==pos]
                        logging.info(f'Length of Subsubset: {len(tmptmp)}')
                        sets.append(tmptmp.copy())
            else:
                for pos in ['before', 'middle', 'after']:
                    tmp = subset[subset.pos==pos]
                    logging.info(f'Length of Subsubset: {len(tmp)}')
                    sets.append(tmp.copy())
            for subsubsubset in sets:
                test = build_from_df(subsubsubset, queries)
                test['query'] = test['query'].apply(preprocess)
                test['text'] = test['text'].apply(preprocess)

                results = scorer(test)
                results.drop_duplicates(inplace=True)

                position = subsubsubset.pos.tolist()[0]
                salience = subsubsubset.salience.tolist()[0]
                
                res = []
                for key, item in lookup_10.items():
                    for doc, score in item.items():
                        adv_score = get_score(key, doc, results)
                        abnirml = ABNIRML(score, adv_score)
                        change = get_rank_change(key, doc, adv_score, lookup_full)
                        res.append({'qid' : key, 'docno' : doc, 'context' : ctx, 'pos' : position, 'salience' : salience, 'adv_score' : adv_score, 'adv_signal' : abnirml, 'rank_change' : change})
                
                frames.append(pd.DataFrame.from_records(res))
    except ValueError:
        pass
                
    pd.concat(frames).to_csv(args.sink)

if __name__ == '__main__':
    args = parser.parse_args()
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
    main(args)