import pyterrier as pt
pt.init()
import argparse
from collections import defaultdict
import logging
import numpy as np
import os
import pandas as pd
import ir_datasets 

from pyterrier_summary.ranker import split_into_sentences
from pyterrier_summary.ranker.neural import SentenceRanker

def count_sentences(text):
    return len(split_into_sentences(text))

def get_random_sentence(text):
    groups = [group for group in split_into_sentences(text)]
    num_sen = len(groups)
    if num_sen <= 1: return text
    return groups[np.random.randint(0, len(groups))]

class Syringe:
    def __init__(self, qrels, ranks) -> None:
        self.ds = ir_datasets.load("msmarco-passage")
        self.docs = pd.DataFrame(self.ds.docs_iter()).set_index('doc_id').text.to_dict()
        self.ranks = ranks
        _qrels = pd.DataFrame(ir_datasets.load(f"msmarco-passage/{qrels}").qrels_iter())
        irrel = _qrels[_qrels['relevance'] < 2]
        rel = _qrels[_qrels['relevance'] >= 2]
        self.qrels = {
            0 : rel,
            1 : irrel,
            2 : rel,
        }
        self.texts = defaultdict(str)
        self.rel = None
        self.pos = None
        self.salient = False

    def _get_position(self, qid, docno):
        order = self.ranks[qid][docno]
        if self.salient == True: return order[0]
        return order[-1]
    
    def _get_text(self, rel, qid):
        qrels = self.qrels[rel]
        if rel == 0: space = qrels[qrels['query_id'] != qid]
        else: space = qrels[qrels['query_id'] == qid]
        text = self.docs[space.sample(1).doc_id.values[0]]
        return get_random_sentence(text)
    
    def _inject(self, target, text, idx):
        adjusted = idx + self.pos
        groups = split_into_sentences(target)
        start = ' '.join(groups[:adjusted])
        end = ' '.join(groups[adjusted:])
        return start +  f' {text} ' + end
    
    def reset_text(self):
        self.texts = defaultdict(str)

    def set_rel(self, rel):
        self.rel = rel

    def set_pos(self, pos):
        self.pos = pos
    
    def set_salient(self, salient):
        self.salient = salient
    
    def inject(self, id, qid):
        assert self.rel is not None
        text = self.docs[id]
        idx = self._get_position(qid, id)
        _text = self.texts[qid]
        if _text != "": payload = _text
        else: 
            payload = self._get_text(self.rel, qid)
            self.texts[qid] = payload

        return self._inject(text, payload, idx)
    
    def transform(self, df, col='adversary'):
        df = df.copy()
        df[col] = df.apply(lambda x : self.inject(x.docno, x.qid), axis=1)
        return df

parser = argparse.ArgumentParser()

parser.add_argument('-model', type=str, default='nq-distilbert-base-v1')
parser.add_argument('-source', type=str)
parser.add_argument('-dataset', type=str)
parser.add_argument('-qrels', type=str)
parser.add_argument('-sink', type=str)

def main(args):

    cols = ['qid', 'docno', 'score']
    types = {'qid' : str, 'docno' : str, 'score' : float}
    ds = ir_datasets.load("msmarco-passage")
    text = pd.DataFrame(ds.docs_iter()).set_index('doc_id').text.to_dict()
    queries = pd.DataFrame(ir_datasets.load(f"msmarco-passage/{args.qrels}").queries_iter()).set_index('query_id').text.to_dict()
    texts = pd.read_csv(args.source, sep='\t', header=None, index_col=False, names=cols, dtype=types)
    inp = []
    for row in texts.itertuples():
        inp.append({'docno':row.docno, 'text':text[row.docno], 'qid':row.qid, 'query': queries[row.qid], 'score':row.score})
    inp = pd.DataFrame.from_records(inp)

    ranker = SentenceRanker(model_name=args.model, setting='ranks')
    ranks = ranker.transform(inp)

    lookup = defaultdict(dict)
    for rank in ranks.itertuples():
        lookup[rank.qid][rank.docno] = rank.summary

    syringe = Syringe(args.qrels, lookup)

    for entity in entities:
        syringe.set_entity(entity)
        for salience in [True, False]:
            syringe.set_salient(salience)
            syringe.set_pos(0)
            before = syringe.transform(texts)
            before['entity'] = entity
            before['rel'] = 'N/A' 
            before['pos'] = 'before'
            before['salience'] = salience_text
            syringe.set_pos(1)

    for rel in [2, 1, 0]:
        syringe.reset_text()
        syringe.set_rel(rel)
        for salience in [True, False]:
            syringe.set_salient(salience)
            salience_text = 'salient' if salience else 'nonsalient'
            logging.info(f'Now computing for {salience_text} sentences injecting rel {rel}...')
            ### BEFORE ### 
            syringe.set_pos(0)
            before = syringe.transform(texts)
            before['rel'] = rel 
            before['pos'] = 'before'
            before['salience'] = salience_text
            ### AFTER ###
            syringe.set_pos(1)
            after = syringe.transform(texts)
            after['rel'] = rel
            after['pos'] = 'after'
            after['salience'] = salience_text
            before.to_csv(os.path.join(args.sink, f'sub.{rel}.{salience_text}.before.tsv'), sep='\t', index=False, header=False)
            after.to_csv(os.path.join(args.sink, f'sub.{rel}.{salience_text}.after.tsv'), sep='\t', index=False, header=False)

if __name__ == '__main__':
    args = parser.parse_args()
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
    main(args)