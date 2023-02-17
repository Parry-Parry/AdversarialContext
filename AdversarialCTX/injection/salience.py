import argparse
from collections import defaultdict
import numpy as np
import os
import pandas as pd
import ir_datasets
from lexrank import LexRank
from lexrank.mappings.stopwords import STOPWORDS

def count_sentences(text):
    return text.count('.')

def get_random_sentence(text):
    groups = [group for group in text.split('.') if len(group.split(' ')) > 4]
    num_sen = len(groups)
    if num_sen == 1: return text
    return groups[np.random.randint(0, len(groups))]

class Syringe:
    def __init__(self, qrels) -> None:
        self.ds = ir_datasets.load("msmarco-passage")
        self.docs = pd.DataFrame(self.ds.docs_iter()).set_index('doc_id').text.to_dict()
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
        self.salient = None
        self.lxr = None

    def _get_position(self, text):
        assert self.lxr is not None, 'Train Lexer!'
        scores = self.lxr(text.split('.'))
        order = np.argmax(scores)
        if self.salient == True: return order[0]
        else: return order[-1]
    
    def _get_text(self, rel, qid):
        qrels = self.qrels[rel]
        if rel == 0: space = qrels[qrels['query_id'] != qid]
        else: space = qrels[qrels['query_id'] == qid]
        text = self.docs[space.sample(1).doc_id.values[0]]
        return get_random_sentence(text)
    
    def _inject(self, target, text, idx):
        adjusted = idx + self.pos
        groups = target.split('.')
        start = '.'.join(groups[:adjusted])
        end = '.'.join(groups[adjusted:])
        return start +  f' {text}. ' + end
    
    def reset_text(self):
        self.texts = defaultdict(str)

    def set_rel(self, rel):
        self.rel = rel

    def set_pos(self, pos):
        self.pos = pos
    
    def set_salient(self, salient):
        self.salient = salient
    
    def inject(self, id, qid):
        idx = None
        text = self.docs[id]
        assert self.rel is not None
        _text = self.texts[qid]
        if _text != "": payload = _text
        else: 
            payload = self._get_text(self.rel, qid)
            self.texts[qid] = payload

        return self._inject(text, payload, idx)
    
    def initialise_lxr(self, docs):
        self.lxr = LexRank(docs, stopwords=STOPWORDS['en'])
    
    def transform(self, df, col='adversary'):
        df = df.copy()
        df[col] = df.apply(lambda x : self.inject(x.docno, x.qid), axis='columns')
        return df
    

parser = argparse.ArgumentParser()

parser.add_argument('-source', type=str)
parser.add_argument('-dataset', type=str)
parser.add_argument('-qrels', type=str)
parser.add_argument('-sink', type=str)

def main(args):
    docs = [doc.text for doc in ir_datasets.load(args.dataset).docs_iter()]
    syringe = Syringe(args.qrels)
    syringe.initialise_lxr(docs)

    cols = ['qid', 'docno', 'score']
    types = {'qid' : str, 'docno' : str, 'score' : float}

    texts = pd.read_csv(args.source, sep='\t', header=None, index_col=False, names=cols, dtype=types)

    for rel in [2, 1, 0]:
        syringe.reset_text()
        syringe.set_rel(rel)
        for salience in [True, False]:
            syringe.set_salient(salience)
            ### BEFORE ### 
            syringe.set_pos(-1)
            end['rel'] = rel 
            end['pos'] = 'before'
            start['salience'] = 'N/A'
            ### AFTER ###
            syringe.set_pos(-1)
            end['rel'] = rel 
            end['pos'] = 'after'
            start['salience'] = 'Salient' if salience else 'Non-Salient'
            start.to_csv(os.path.join(args.sink, f'sub.{rel}.start.tsv'), sep='\t', index=False, header=False)
            mid.to_csv(os.path.join(args.sink, f'sub.{rel}.mid.tsv'), sep='\t', index=False, header=False)
            end.to_csv(os.path.join(args.sink, f'sub.{rel}.end.tsv'), sep='\t', index=False, header=False)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)