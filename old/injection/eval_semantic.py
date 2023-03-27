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
from sklearn.metrics import pairwise_distances
import sentence_transformers

### BEGIN EVAL MODELS ###

class Encoder:
    def __init__(self, model_id, gpu=False) -> None:
        from sentence_transformers import SentenceTransformer
        if device is None:
            device = 'cuda' if gpu else 'cpu'
        self.device = torch.device(device)       
        self.model = SentenceTransformer(model_id, device=self.device)

    def embedding(self, text):
        return self.model.encode([text], convert_to_numpy=True)[0]
    
    def compare(self, q, d1, d2):
        q_enc = self.embedding(q)
        d1_enc = self.embedding(d1)
        d2_enc = self.embedding(d2)

        dist = pairwise_distances(q_enc.reshape(1, -1), np.stack([d1_enc, d2_enc], axis=0), metric='cosine')[0]
        return dist[0] - dist[1]

def clean_text(text):
    text = re.sub(r'[^A-Za-z0-9 ]+', '', text)
    return re.sub(r'/[^\x00-\x7F]/g', '', text).strip()

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

def compile_lookup(df):
    lookup = defaultdict(dict)
    for row in df.itertuples():
        lookup[row.qid][row.docno]=row.adversary
    return lookup
### END CONVIENIENCE FUNCTIONS ###

parser = argparse.ArgumentParser()

parser.add_argument('-ctxsource', type=str)
parser.add_argument('-statsource', type=str)
parser.add_argument('-embedding_model', type=str)
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

    semantic_distance = Encoder(args.embedding_model, args.gpu)

    def get_diff(qid, docno, adv, lookup):
        static = lookup[qid][docno]
        query = queries[qid]
        return semantic_distance.compare(query, adv, static)
    ### END LOOKUPS AND MODELS INIT ###

    cols = ['qid', 'docno', 'score', 'adversary', 'rel', 'pos', 'salience', 'salience_type', 'sentence', 'context']
    types = {'qid' : str, 'docno' : str, 'score' : float, 'adversary' : str, 'rel' : str, 'pos':str, 'salience':str, 'salience_type':str, 'sentence':str, 'context':str}

    frames = []
    ctxtext = pd.read_csv(args.ctxsource, header=None, index_col=False, names=cols, dtype=types, on_bad_lines='skip')
    statictext = pd.read_csv(args.staticsource, header=None, index_col=False, names=cols, dtype=types, on_bad_lines='skip')
    for ctx in ctxtext.context.unique().tolist():
        ctxsubset = ctxtext[ctxtext.context==ctx]
        staticsubset = statictext[statictext.context==ctx]
        sets  = []
        if args.type == 'salience': # Very very very very ugly
            for sal in ['salient', 'nonsalient']:
                ctxtmp = ctxsubset[ctxsubset.salience==sal]
                statictmp = staticsubset[staticsubset.salience==sal]
                for pos in ['before', 'after']:
                    ctxtmptmp = ctxtmp[ctxtmp.pos==pos]
                    statictmptmp = statictmp[statictmp.pos==pos]
                    sets.append((ctxtmptmp.copy(), statictmptmp.copy()))
        else:
            for pos in ['before', 'middle', 'after']:
                ctxtmp = ctxsubset[ctxsubset.pos==pos]
                statictmp = staticsubset[staticsubset.pos==pos]
                sets.append((ctxtmp.copy(), statictmp.copy()))
        for ctxset, staticset in sets:
            static_rows = compile_lookup(staticset)
            ctxset['semantic_diff'] = ctxset.apply(lambda x : get_diff(x['qid'], x['docno'], x['adversary'], static_rows[ctx]), axis=1)
            frames.append(ctxset)
    pd.concat(frames).to_csv(args.sink)

if __name__ == '__main__':
    args = parser.parse_args()
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
    main(args)