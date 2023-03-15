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
from pyterrier_freetext.summary.ranker.lexrank import LexRanker

SENTENCE_MODEL = 'nq-distilbert-base-v1'
T5_MODEL = None

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
    
    def inject(self, id, qid, payload):
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
        sentences = f.readlines()

    ds = ir_datasets.load("msmarco-passage")
    text = pd.DataFrame(ds.docs_iter()).set_index('doc_id').text.to_dict()
    queries = pd.DataFrame(ir_datasets.load(f"msmarco-passage/{args.qrels}").queries_iter()).set_index('query_id').text.to_dict()

    cols = ['qid', 'docno', 'score']
    types = {'qid' : str, 'docno' : str, 'score' : float}
    texts = pd.read_csv(args.source, sep='\t', header=None, index_col=False, names=cols, dtype=types)

    inp = []
    for row in texts.itertuples():
        inp.append({'docno':row.docno, 'text':text[row.docno], 'qid':row.qid, 'query': queries[row.qid], 'score':row.score})
    inp = pd.DataFrame.from_records(inp)

    lookups = {}

    sentenceranker = SentenceRanker(model_name=SENTENCE_MODEL, setting='ranks')

    ranks = sentenceranker.transform(inp)
    lookups['sentence'] = defaultdict(dict)
    for rank in ranks.itertuples():
        lookups['sentence'][rank.qid][rank.docno] = rank.summary

    t5ranker = MonoT5SentenceRanker(setting='ranks')

    ranks = t5ranker.transform(inp)
    lookups['t5'] = defaultdict(dict)
    for rank in ranks.itertuples():
        lookups['t5'][rank.qid][rank.docno] = rank.summary

    ds = pt.get_dataset('msmarco_passage')
    index = ds.get_index(variant='terrier_stemmed')
    lexranker = LexRanker(setting='ranks', background_index=index)

    ranks = lexranker.transform(inp)
    lookups['lexrank'] = defaultdict(dict)
    for rank in ranks.itertuples():
        lookups['lexrank'][rank.qid][rank.docno] = rank.summary

    
    for name, lookup in lookups.items():
        syringe = Syringe(lookup)
        afters = []
        befores = []
        for sentence in sentences:
            for salience in [True, False]:
                syringe.set_salient(salience)
                salience_text = 'salient' if salience else 'nonsalient'

                ### BEFORE ### 

                syringe.set_pos(0)
                before = syringe.transform(texts)
                before['rel'] = 'NA' 
                before['pos'] = 'before'
                before['salience'] = salience_text
                before['salience_type'] = name
                before['sentence'] = sentence
                
                ### AFTER ###

                syringe.set_pos(1)
                after = syringe.transform(texts)
                after['rel'] = 'NA'
                after['pos'] = 'after'
                after['salience'] = salience_text
                after['salience_type'] = name
                after['sentence'] = sentence

                afters.append(after)
                befores.append(before)
        pd.concat(afters).to_csv(os.path.join(args.sink, f'{name}.after.csv'))
        pd.concat(befores).to_csv(os.path.join(args.sink, f'{name}.before.csv'))
            


if __name__ == '__main__':
    args = parser.parse_args()
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
    main(args)