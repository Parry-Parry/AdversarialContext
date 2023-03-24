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
    def __init__(self, ranks) -> None:
        self.ds = ir_datasets.load("msmarco-passage")
        self.docs = pd.DataFrame(self.ds.docs_iter()).set_index('doc_id').text.to_dict()
        self.ranks = ranks
        
        self.pos = None
        self.salient = False
    
    def _get_position(self, qid, docno):
        order = self.ranks[qid][docno]
        if self.salient == True: return order[0]
        return order[-1]
    
    def _inject(self, target, text, idx):
        adjusted = idx + self.pos
        groups = split_into_sentences(target)
        start = ' '.join(groups[:adjusted])
        end = ' '.join(groups[adjusted:])
        return start +  f' {text} ' + end

    def set_pos(self, pos):
        self.pos = pos
    
    def set_salient(self, salient):
        self.salient = salient
    
    def inject(self, id, qid, text):
        payload = text
        text = self.docs[id]
        idx = self._get_position(qid, id)

        return self._inject(text, payload, idx)
    
    def transform(self, df, text, col='adversary'):
        df = df.copy()
        df[col] = df.apply(lambda x : self.inject(x.docno, x.qid, text), axis=1)
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

    cols = ['qid', 'docno']
    types = {'qid' : str, 'docno' : str}
    texts = pd.read_csv(args.source, sep='\t', header=None, index_col=False, names=cols, dtype=types)

    inp = []
    for row in texts.itertuples():
        inp.append({'docno':row.docno, 'text':text[row.docno], 'qid':row.qid, 'query': queries[row.qid]})
    inp = pd.DataFrame.from_records(inp)

    lookups = {}

    sentenceranker = SentenceRanker(model_name=SENTENCE_MODEL, mode='ranks')

    ranks = sentenceranker.transform(inp)
    lookups['sentence'] = defaultdict(dict)
    for rank in ranks.itertuples():
        lookups['sentence'][rank.qid][rank.docno] = rank.summary

    t5ranker = MonoT5SentenceRanker(model_name=None, mode='ranks')

    ranks = t5ranker.transform(inp)
    lookups['t5'] = defaultdict(dict)
    for rank in ranks.itertuples():
        lookups['t5'][rank.qid][rank.docno] = rank.summary

    for c, s in zip(ctx, sentences):
        s = s.strip('\n')
        c = c.strip('\n')
        for name, lookup in lookups.items():
            frames = []
            for c in set(ctx):
                syringe = Syringe(lookup)
                for salience in [True, False]:
                    syringe.set_salient(salience)
                    salience_value = 'salient' if salience else 'nonsalient'
                    ### BEFORE ### 
                    syringe.set_pos(0)
                    before = syringe.transform(texts, s)
                    before['rel'] = 'NA' 
                    before['pos'] = 'before'
                    before['salience'] = salience_value
                    before['salience_type'] = name
                    before['sentence'] = s
                    before['context'] = c
                    
                    ### AFTER ###

                    syringe.set_pos(1)
                    after = syringe.transform(texts, s)
                    after['rel'] = 'NA'
                    after['pos'] = 'after'
                    after['salience'] = salience_value
                    after['salience_type'] = name
                    after['sentence'] = s
                    after['context'] = c

                    frames.append(after)
                    frames.append(before)
            pd.concat(frames).to_csv(os.path.join(args.sink, f'static.{name}.csv'), index=False, header=False)

if __name__ == '__main__':
    args = parser.parse_args()
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
    main(args)