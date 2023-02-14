import argparse
from collections import defaultdict
import numpy as np
import os
import pandas as pd
import ir_datasets

def count_sentences(text):
    return text.count('.')

def get_random_sentence(text):
    groups = text.split('.')
    num_sen = len(groups)
    if num_sen == 1: return text
    return f'{groups[np.random.randint(0, len(groups))]}.'

class Syringe:
    def __init__(self, qrels) -> None:
        self.ds = ir_datasets.load("irds:msmarco-passage")
        self.docs = pd.DataFrame(self.ds.docs_iter()).set_index('doc_id').text.to_dict()
        _qrels = pd.DataFrame(ir_datasets.load(f"irds:msmarco-passage/{qrels}").qrels_iter()).rename({'query_id':'qid', 'relevance':'rel'})
        self.qrels = {
            0 : _qrels[_qrels.rel == 0],
            1 : _qrels[_qrels.rel == 1],
            2 : _qrels[_qrels.rel == 2]
        }
        self.texts = defaultdict(str)
        self.rel = None
        self.pos = None
    
    def _get_text(self, rel, qid):
        _text = self.texts[qid]
        if _text != "": return _text
        qrels = self.qrels[rel]
        text = self.docs[qrels[qrels.qid == qid].sample(1).doc_id]
        return get_random_sentence(text)
    
    def _inject(self, target, text, pos):
        if pos == 0: return f'{text} {target}'
        if pos == -1: return f'{target} {text}'
        groups = target.split('.')
        start = groups[:pos].join('.')
        end = groups[pos:].join('.')
        return f'{start} {text} {end}'
    
    def _reset_text(self):
        self.texts = defaultdict(str)

    def set_rel(self, rel):
        self.rel = rel

    def set_pos(self, pos):
        self.pos = pos
    
    def inject(self, id, qid):
        text = self.docs[id]
        assert self.rel is not None
        payload = self._get_text(self.rel, qid)
        if self.pos not in [0, -1]: pos = count_sentences(text) // 2
        else: pos = self.pos

        return self._inject(text, payload, pos)
    
    def transform(self, df, col='adversary'):
        self._reset_text()
        df = df.copy()
        df[col] = df.apply(lambda x : self._inject(x.docno, x.qid), axis='columns')
        return df
    

parser = argparse.ArgumentParser()

parser.add_argument('-source', type=str)
parser.add_argument('-qrels', type=str)
parser.add_argument('-sink', type=str)

def main(args):
    syringe = Syringe(args.qrels)

    cols = ['qid', 'docno', 'score']
    types = {'qid' : str, 'docno' : str, 'score' : float}

    texts = pd.read_csv(args.source, sep='\t', header=None, index_col=False, names=cols, dtype=types)

    for rel in [2, 1, 0]:
        syringe.set_rel(rel)
        ## START ##
        syringe.set_pos(0)
        start = syringe.transform(texts)
        ## MIDDLE ##
        syringe.set_pos(1)
        mid = syringe.transform(texts)
        ## END ## 
        syringe.set_pos(-1)
        end = syringe.transform(texts)

        start.to_csv(os.path.join(args.sink, f'sub.{rel}.start.tsv'), sep='\t', index=False, header=False)
        mid.to_csv(os.path.join(args.sink, f'sub.{rel}.mid.tsv'), sep='\t', index=False, header=False)
        end.to_csv(os.path.join(args.sink, f'sub.{rel}.end.tsv'), sep='\t', index=False, header=False)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)