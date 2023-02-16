import argparse
from collections import defaultdict
import numpy as np
import os
import pandas as pd
import ir_datasets

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
    
    def _get_text(self, rel, qid):
        qrels = self.qrels[rel]
        if rel == 0: space = qrels[qrels['query_id'] != qid]
        else: space = qrels[qrels['query_id'] == qid]
        text = self.docs[space.sample(1).doc_id.values[0]]
        return get_random_sentence(text)
    
    def _inject(self, target, text, pos):
        if pos == 0: return f'{text}. ' + target
        if pos == -1: return target + f' {text}'
        groups = target.split('.')
        start = '.'.join(groups[:pos])
        end = '.'.join(groups[pos:])
        return start +  f' {text}. ' + end
    
    def reset_text(self):
        self.texts = defaultdict(str)

    def set_rel(self, rel):
        self.rel = rel

    def set_pos(self, pos):
        self.pos = pos
    
    def inject(self, id, qid):
        text = self.docs[id]
        assert self.rel is not None
        _text = self.texts[qid]
        if _text != "": payload = _text
        else: 
            payload = self._get_text(self.rel, qid)
            self.texts[qid] = payload
        if self.pos not in [0, -1]: pos = count_sentences(text) // 2
        else: pos = self.pos

        return self._inject(text, payload, pos)
    
    def transform(self, df, col='adversary'):
        df = df.copy()
        df[col] = df.apply(lambda x : self.inject(x.docno, x.qid), axis='columns')
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
        syringe.reset_text()
        syringe.set_rel(rel)
        ## START ##
        syringe.set_pos(0)
        start = syringe.transform(texts)
        start['rel'] = rel 
        start['pos'] = 'start'
        ## MIDDLE ##
        syringe.set_pos(1)
        mid = syringe.transform(texts)
        mid['rel'] = rel 
        mid['pos'] = 'mid'
        ## END ## 
        syringe.set_pos(-1)
        end = syringe.transform(texts)
        end['rel'] = rel 
        end['pos'] = 'end'

        start.to_csv(os.path.join(args.sink, f'sub.{rel}.start.tsv'), sep='\t', index=False, header=False)
        mid.to_csv(os.path.join(args.sink, f'sub.{rel}.mid.tsv'), sep='\t', index=False, header=False)
        end.to_csv(os.path.join(args.sink, f'sub.{rel}.end.tsv'), sep='\t', index=False, header=False)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)