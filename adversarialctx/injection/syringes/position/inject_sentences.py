import pyterrier as pt
pt.init()
import argparse
from collections import defaultdict
import logging
import numpy as np
import os
import pandas as pd
import ir_datasets 

from pyterrier_freetext.util import split_into_sentences
from pyterrier_freetext.summary.ranker import SentenceRanker, MonoT5SentenceRanker

SENTENCE_MODEL = 'nq-distilbert-base-v1'

def count_sentences(text):
    return len(split_into_sentences(text))

def get_random_sentence(text):
    groups = [group for group in split_into_sentences(text)]
    num_sen = len(groups)
    if num_sen <= 1: return text
    return groups[np.random.randint(0, len(groups))]

class Syringe:
    def __init__(self) -> None:
        self.ds = ir_datasets.load("msmarco-passage")
        self.docs = pd.DataFrame(self.ds.docs_iter()).set_index('doc_id').text.to_dict()
        
        self.pos = None
    
    def _inject(self, target, text):
        if self.pos == 0: return f'{text}. ' + target
        if self.pos == -1: return target + f' {text}'
        
        groups = split_into_sentences(target)
        pos = len(groups) // 2 if len(groups) > 1 else 0
        adjusted = pos + 1
        start = ' '.join(groups[:adjusted])
        end = ' '.join(groups[adjusted:])
        return start +  f' {text} ' + end

    def set_pos(self, pos):
        self.pos = pos
    
    def inject(self, id, text):
        payload = text
        text = self.docs[id]
        return self._inject(text, payload)
    
    def transform(self, df, text, col='adversary'):
        df = df.copy()
        df[col] = df.apply(lambda x : self.inject(x.docno, text), axis=1)
        return df

parser = argparse.ArgumentParser()

parser.add_argument('-sentence_source', type=str)
parser.add_argument('-source', type=str)
parser.add_argument('-dataset', type=str)
parser.add_argument('-sink', type=str)

def main(args):
    with open(args.sentence_source, 'r') as f:
        text_items = map(lambda x : x.split('\t'), f.readlines())

    ctx, sentences =  map(list, zip(*text_items))
    
    ds = ir_datasets.load("msmarco-passage")
    text = pd.DataFrame(ds.docs_iter()).set_index('doc_id').text.to_dict()
    queries = pd.DataFrame(ir_datasets.load(f"msmarco-passage/{args.dataset}").queries_iter()).set_index('query_id').text.to_dict()

    cols = ['qid', 'docno', 'score']
    types = {'qid' : str, 'docno' : str, 'score' : float}
    texts = pd.read_csv(args.source, sep='\t', header=None, index_col=False, names=cols, dtype=types)

    inp = []
    for row in texts.itertuples():
        inp.append({'docno':row.docno, 'text':text[row.docno], 'qid':row.qid, 'query': queries[row.qid], 'score':row.score})
    inp = pd.DataFrame.from_records(inp)

    afters, befores, middles = [], [], []

    syringe = Syringe()
    for c, s in zip(ctx, sentences):
        salience_value = 'NA'
        ### BEFORE ### 
        syringe.set_pos(0)
        before = syringe.transform(texts, s)
        before['rel'] = 'NA' 
        before['pos'] = 'before'
        before['salience'] = salience_value
        before['salience_type'] = 'NA'
        before['sentence'] = s
        before['context'] = c

        ### MIDDLE ### 
        syringe.set_pos(1)
        middle = syringe.transform(texts, s)
        middle['rel'] = 'NA' 
        middle['pos'] = 'middle'
        middle['salience'] = salience_value
        middle['salience_type'] = 'NA'
        middle['sentence'] = s
        middle['context'] = c
        
        ### AFTER ###

        syringe.set_pos(-1)
        after = syringe.transform(texts, s)
        after['rel'] = 'NA'
        after['pos'] = 'after'
        after['salience'] = salience_value
        after['salience_type'] = 'NA'
        after['sentence'] = s
        after['context'] = c

        afters.append(after)
        middles.append(middle)
        befores.append(befores)

    pd.concat(afters).to_csv(os.path.join(args.sink, f'after.csv'), index=False, header=False)
    pd.concat(middles).to_csv(os.path.join(args.sink, f'middle.csv'), index=False, header=False)
    pd.concat(befores).to_csv(os.path.join(args.sink, f'before.csv'), index=False, header=False)
            


if __name__ == '__main__':
    args = parser.parse_args()
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
    main(args)