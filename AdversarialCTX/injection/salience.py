import argparse
import bz2
from collections import defaultdict
import logging
import multiprocessing as mp
import pickle
import numpy as np
import os
import pandas as pd
import ir_datasets
from lexrank import LexRank
from lexrank.mappings.stopwords import STOPWORDS
import re 
from pandarallel import pandarallel

### Sentence Regex from: https://stackoverflow.com/a/31505798

alphabets= "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov|edu|me)"
digits = "([0-9])"

def split_into_sentences(text):
    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    text = re.sub(digits + "[.]" + digits,"\\1<prd>\\2",text)
    if "..." in text: text = text.replace("...","<prd><prd><prd>")
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    return sentences

def count_sentences(text):
    return len(split_into_sentences(text))

def get_random_sentence(text):
    groups = [group for group in split_into_sentences(text)]
    num_sen = len(groups)
    if num_sen == 1: return text
    return groups[np.random.randint(0, len(groups))]

def extract_text(item): 
    return item.text

class Syringe:
    def __init__(self, qrels, threads=4) -> None:
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
        self.threads = threads
        self.texts = defaultdict(str)
        self.rel = None
        self.pos = None
        self.salient = None
        self.lxr = None

    def _get_position(self, text):
        assert self.lxr is not None, 'Train Lexer!'
        scores = self.lxr.rank_sentences(split_into_sentences(text), fast_power_method=True)
        order = np.argsort(scores)
        if self.salient == True: return order[-1]
        return order[0]
    
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
        idx = self._get_position(text)
        _text = self.texts[qid]
        if _text != "": payload = _text
        else: 
            payload = self._get_text(self.rel, qid)
            self.texts[qid] = payload

        return self._inject(text, payload, idx)
    
    def initialise_lxr(self, docs):
        logging.info('Initialising Lexer...')
        self.lxr = LexRank(docs, stopwords=STOPWORDS['en'])
        logging.info('Done!')
    
    def transform(self, df, col='adversary'):
        df = df.copy()
        df[col] = df.apply(lambda x : self.inject(x.docno, x.qid), axis=1)
        return df

parser = argparse.ArgumentParser()

parser.add_argument('-source', type=str)
parser.add_argument('-salience', type=str)
parser.add_argument('-dataset', type=str)
parser.add_argument('-qrels', type=str)
parser.add_argument('-sink', type=str)

parser.add_argument('--threads', type=int, default=4)

def main(args):
    with bz2.BZ2File(args.salience, 'rb') as f:
        split_docs = pickle.load(f)

    syringe = Syringe(args.qrels)
    syringe.initialise_lxr(split_docs)

    cols = ['qid', 'docno', 'score']
    types = {'qid' : str, 'docno' : str, 'score' : float}

    texts = pd.read_csv(args.source, sep='\t', header=None, index_col=False, names=cols, dtype=types)

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