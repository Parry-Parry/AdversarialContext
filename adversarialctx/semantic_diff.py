import pyterrier as pt
pt.init()
from collections import defaultdict
import logging
import torch
import argparse
import os
import ir_datasets
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

### BEGIN CONVIENIENCE FUNCTIONS ###

def compile_lookup(df):
    lookup = defaultdict(dict)
    for row in df.itertuples():
        lookup[row.qid][row.docno]=row.adversary
    return lookup

def read_frame(f, c):
    with open(f, 'r') as f:
        text_items = map(lambda x : [y.strip('\n') for y in x.split('\t')], f.readlines())
    v = list(map(list, zip(*text_items)))
    return pd.DataFrame.from_dict({r : v for r, v in zip(c, v)})

### END CONVIENIENCE FUNCTIONS ###
parser = argparse.ArgumentParser()

parser.add_argument('-ctxsource', type=str)
parser.add_argument('-staticsource', type=str)
parser.add_argument('-full_path', type=str)
parser.add_argument('-embedding_model', type=str)
parser.add_argument('-qrels', type=str)
parser.add_argument('-sink', type=str)

parser.add_argument('--gpu', action='store_true')

def main(args):

    ### BEGIN LOOKUPS AND MODELS INIT ###
    ds = ir_datasets.load(args.qrels)
    queries = pd.DataFrame(ds.queries_iter()).set_index('query_id').text.to_dict()

    semantic_distance = Encoder(args.embedding_model, args.gpu)

    def get_diff(qid, context, static):
        query = queries[qid]
        return semantic_distance.compare(query, context, static)

    format_10 = f'{args.scorer}.10.tsv'
    lookup_10 = defaultdict(dict)
    with open(os.path.join(args.full_path, format_10), 'r') as f:
        items = map(lambda x : x.split('\t'), f.readlines())
    
    for item in items: lookup_10[item[0]][item[1]] = float(item[2].strip())

    ### END LOOKUPS AND MODELS INIT ###

    cols = ['qid', 'docno', 'adversary', 'rel', 'pos', 'salience', 'salience_type', 'sentence', 'context']
    ctxcols = ['qid', 'docno', 'adversary', 'sentence', 'rel', 'pos', 'salience', 'salience_type', 'context']
   
    frames = []

    ctxtext = read_frame(args.ctxsource, ctxcols)
    statictext = read_frame(args.staticsource, cols)

    filtercol = lambda x, y, z : x[x[y]==z].copy()

    for ctx in ctxtext.context.unique().tolist():
            ctxsub = filtercol(ctxtext, 'context', ctx)
            staticsub = filtercol(statictext, 'context', ctx)
            sets = []
            if args.type == 'salience':
                for sal in ['salient', 'nonsalient']:
                    ctxsubsub = filtercol(ctxsub, 'salience', sal)
                    staticsubsub = filtercol(staticsub, 'salience', sal)
                    for pos in ['before', 'after']:
                        sets.append((filtercol(ctxsubsub, 'pos', pos), filtercol(staticsubsub, 'pos', pos)))
            else:
                for pos in ['before', 'middle', 'after']:
                    sets.append((filtercol(ctxsub, 'pos', pos), filtercol(staticsub, 'pos', pos)))
            for c, s in sets:
                position = c.pos.tolist()[0]
                salience = c.salience.tolist()[0]

                ctxlookup = compile_lookup(c)
                staticlookup = compile_lookup(s)
                
                res = []
                for key, item in lookup_10.items():
                    for doc, _ in item.items():
                        diff = get_diff(key, ctxlookup[key][doc], staticlookup[key][doc])
                        res.append({'qid' : key, 'docno' : doc, 'context' : ctx, 'pos' : position, 'salience' : salience, 'semantic_difference' : diff})
                
                frames.append(pd.DataFrame.from_records(res))
    pd.concat(frames).to_csv(args.sink)

if __name__ == '__main__':
    args = parser.parse_args()
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
    main(args)