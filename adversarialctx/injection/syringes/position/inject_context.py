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

def count_sentences(text):
    return len(split_into_sentences(text))

def get_random_sentence(text):
    groups = [group for group in split_into_sentences(text)]
    num_sen = len(groups)
    if num_sen <= 1: return text
    return groups[np.random.randint(0, len(groups))]

class Syringe:
    def __init__(self, lookup) -> None:
        self.ds = ir_datasets.load("msmarco-passage")
        self.docs = pd.DataFrame(self.ds.docs_iter()).set_index('doc_id').text.to_dict()
        
        self.ctx = None
        self.lookup = lookup
        self.pos = None
    
    def set_ctx(self, ctx):
        self.ctx = ctx

    def get_payload(self, id, qid):
        return self.lookup[self.ctx][qid][id]

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
    
    def inject(self, id, qid, text):
        payload = self.get_payload(id, qid)
        text = self.docs[id]
        return self._inject(text, payload)
    
    def transform(self, df, text, col='adversary'):
        df = df.copy()
        df[col] = df.apply(lambda x : self.inject(x.docno, x.qid,  text), axis=1)
        df['sentence'] = df.apply(lambda x : self.get_payload(x.docno, x.qid), axis=1)
        return df

parser = argparse.ArgumentParser()

parser.add_argument('-sentence_source', type=str)
parser.add_argument('-source', type=str)
parser.add_argument('-dataset', type=str)
parser.add_argument('-sink', type=str)

def main(args):
    with open(args.sentence_source, 'r') as f:
        text_items = map(lambda x : x.split('\t'), f.readlines())

    ctx, qidx, docnos, sentences =  map(list, zip(*text_items))

    cols = ['qid', 'docno']
    types = {'qid' : str, 'docno' : str}
    texts = pd.read_csv(args.source, sep='\t', header=None, index_col=False, names=cols, dtype=types)

    dictlook = {}
    for c in set(ctx):
        dictlook[c] = defaultdict(dict)
        
    for item in zip(ctx, qidx, docnos, sentences):
        c, q, d, s = item
        try:
            dictlook[c][q][d] = s.strip()
        except KeyError:
            pass


    frames = []
    syringe = Syringe(dictlook)
    for c in set(ctx):
        syringe.set_ctx(c)
        
        salience_value = 'NA'
        ### BEFORE ### 
        syringe.set_pos(0)
        before = syringe.transform(texts)
        before['rel'] = 'NA' 
        before['pos'] = 'before'
        before['salience'] = salience_value
        before['salience_type'] = 'NA'
        before['sentence'] = s
        before['context'] = c

        ### MIDDLE ### 
        syringe.set_pos(1)
        middle = syringe.transform(texts)
        middle['rel'] = 'NA' 
        middle['pos'] = 'middle'
        middle['salience'] = salience_value
        middle['salience_type'] = 'NA'
        middle['sentence'] = s
        middle['context'] = c
        
        ### AFTER ###

        syringe.set_pos(-1)
        after = syringe.transform(texts)
        after['rel'] = 'NA'
        after['pos'] = 'after'
        after['salience'] = salience_value
        after['salience_type'] = 'NA'
        after['sentence'] = s
        after['context'] = c

        frames.append(after)
        frames.append(middle)
        frames.append(before)
    pd.concat(frames).to_csv(os.path.join(args.sink, f'positional.context.csv'), index=False, header=False)
            

if __name__ == '__main__':
    args = parser.parse_args()
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
    main(args)