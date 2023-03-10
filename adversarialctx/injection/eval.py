import logging
import pyterrier as pt
pt.init()
import argparse
import os
import ir_datasets
from typing import NamedTuple
import re
import pandas as pd

class cfg(NamedTuple):
    name : str
    dataset : str
    checkpoint : str 
    gpu : bool

def clean_text(text):
    text = re.sub(r'[^A-Za-z0-9 ]+', '', text)
    return re.sub(r'/[^\x00-\x7F]/g', '', text).strip()

def init_colbert(config):
    from pyterrier_colbert.ranking import ColBERTModelOnlyFactory
    pytcolbert = ColBERTModelOnlyFactory(config.checkpoint, gpu=config.gpu)
    preprocess = lambda x : x
    return pytcolbert.text_scorer(), preprocess

def init_dr(config):
    from pyterrier_dr import ElectraScorer, TasB
    if config.name == 'electra': 
        scorer = ElectraScorer()
    else: 
        assert config.checkpoint is not None
        scorer = TasB(config.checkpoint) 

    preprocess = lambda x : x
    return scorer, preprocess
    
def init_monot5(config):
    from pyterrier_t5 import MonoT5ReRanker
    preprocess = lambda x : x
    return MonoT5ReRanker(), preprocess

def init_bm25(config):
    ds = pt.get_dataset(config.dataset)
    indx = pt.IndexFactory.of(ds.get_index(variant='terrier_stemmed'))
    scorer = pt.batchretrieve.TextScorer(body_attr='text', wmodel='BM25', background_index=indx, properties={"termpipelines" : "Stopwords,PorterStemmer"})
    return scorer, clean_text

_logger = ir_datasets.log.easy()

def build_data(path):
  result = []
  dataset = ir_datasets.load(path)
  docs = dataset.docs_store()
  queries = {q.query_id: q.text for q in dataset.queries_iter()}
  for qrel in _logger.pbar(ir_datasets.load(path).scoreddocs_iter(), desc='dev data'):
    if qrel.query_id in queries:
      result.append([qrel.query_id, queries[qrel.query_id], qrel.doc_id, docs.get(qrel.doc_id).text])
  return pd.DataFrame(result, columns=['qid', 'query', 'docno', 'text'])

scorers = {
    'tasb' : init_dr,
    'electra' : init_dr,
    't5' : init_monot5,
    'colbert' : init_colbert,
    'bm25' : init_bm25
}

parser = argparse.ArgumentParser()

parser.add_argument('-source', type=str)
parser.add_argument('-scorer', type=str)
parser.add_argument('-qrels', type=str)
parser.add_argument('-sink', type=str)

parser.add_argument('--dataset', type=str, default=None)
parser.add_argument('--checkpoint', type=str, default=None)
parser.add_argument('--gpu', action='store_true')

def get_rank_change(qid, text):
   pass

def main(args):
    ds = ir_datasets.load(args.qrels)
    queries = pd.DataFrame(ds.queries_iter()).set_index('query_id').text.to_dict()

    config = cfg(args.scorer, args.dataset, args.checkpoint, args.gpu)

    try:
        scorer, preprocess = scorers[args.scorer](config)
    except KeyError:
        logging.error(f'Model {args.scorer} not recognised!')
        exit

    def build_from_df(df):
        new = []
        for row in df.itertuples():
            new.append({'qid':row.qid, 'query':queries[row.qid], 'docno':row.docno, 'text':row.adversary})
        return pd.DataFrame.from_records(new)

    cols = ['qid', 'docno', 'score', 'adversary', 'rel', 'pos', 'salience']
    types = {'qid' : str, 'docno' : str, 'score' : float, 'adversary' : str, 'rel' : int, 'pos':str, 'salience':str}

    advers = [f for f in os.listdir(args.source) if os.path.isfile(os.path.join(args.source, f))]

    frames = []
    for text in advers:
      texts = pd.read_csv(os.path.join(args.source, text), sep='\t', header=None, index_col=False, names=cols, dtype=types)
      test = build_from_df(texts)
      test['query'] = test['query'].apply(preprocess)
      test['text'] = test['text'].apply(preprocess)
      results = scorer(test)

      def ABNIRML(qid, docno, score):
        tmp = results[results['qid']==qid].set_index('docno')['score']
        adv_score = tmp.loc[docno]
        diff = score - adv_score
        if diff < 0: return -1 
        elif diff > 0: return 1
        return 0

      def get_score(qid, docno):
        tmp = results[results['qid']==qid].set_index('docno')['score']
        return tmp.loc[docno]

      texts['adv_signal'] = texts.apply(lambda x : ABNIRML(x['qid'], x['docno'], x['score']), axis=1)
      texts['adv_score'] = texts.apply(lambda x : get_score(x['qid'], x['docno']), axis=1)
      frames.append(texts)

    out = pd.concat(frames)
    out.to_csv(args.sink)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)