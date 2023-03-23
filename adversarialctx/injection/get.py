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
    index_path : str 
    index_name : str
    gpu : bool

def clean_text(text):
    text = re.sub(r'[^A-Za-z0-9 ]+', '', text)
    return re.sub(r'/[^\x00-\x7F]/g', '', text).strip()

def init_colbert(config):
    from pyterrier_colbert.ranking import ColBERTFactory
    pytcolbert = ColBERTFactory(config.checkpoint, config.index_path, config.index_name)
    pytcolbert.faiss_index_on_gpu = config.gpu 
    preprocess = lambda x : x
    return pytcolbert.end_to_end(), preprocess

def init_dr(config):
    from pyterrier_dr import ElectraScorer, TasB, NumpyIndex
    if config.name == 'electra': 
        dataset = pt.get_dataset(config.dataset)
        bm25 = pt.BatchRetrieve.from_dataset(config.dataset, 'terrier_stemmed_text', wmodel='BM25', metadata=['docno', 'text'])
        model = ElectraScorer()
        scorer = bm25 >> pt.text.get_text(dataset.get_index(variant='terrier_stemmed_text'), "text") >> model
        preprocess = clean_text
    else: 
        assert config.checkpoint is not None
        assert config.index_path is not None
        index = NumpyIndex(config.index_path)
        model = TasB(config.checkpoint) 
        scorer = model >> index
        preprocess = lambda x : x
    
    return scorer, preprocess
    
def init_monot5(config):
    from pyterrier_t5 import MonoT5ReRanker
    dataset = pt.get_dataset(config.dataset)
    bm25 = pt.BatchRetrieve.from_dataset(config.dataset, 'terrier_stemmed_text', wmodel='BM25', metadata=['docno', 'text'])
    monoT5 = MonoT5ReRanker()
    return bm25 >> pt.text.get_text(dataset.get_index(variant='terrier_stemmed_text'), 'text') >> monoT5, clean_text

def init_bm25(config):
    scorer = pt.BatchRetrieve.from_dataset(config.dataset, 'terrier_stemmed_text', wmodel='BM25', metadata=['docno', 'text'])
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

parser.add_argument('-scorer', type=str)
parser.add_argument('-qrels', type=str)
parser.add_argument('-topk', type=int)
parser.add_argument('-sink', type=str)

parser.add_argument('--dataset', type=str, default=None)
parser.add_argument('--checkpoint', type=str, default=None)
parser.add_argument('--index_path', type=str, default=None)
parser.add_argument('--index_name', type=str, default=None)
parser.add_argument('--gpu', action='store_true')

def main(args):
    data = build_data(args.qrels)
    queries = data[['qid', 'query']].copy().drop_duplicates()

    config = cfg(args.scorer, args.dataset, args.checkpoint, args.index_path, args.index_name, args.gpu)

    try:
        scorer, preprocess = scorers[args.scorer](config)
    except KeyError:
        logging.error(f'Model {args.scorer} not recognised!')
        exit

    scorer = scorer % args.topk

    queries['query'] = queries['query'].apply(preprocess)
    topk = scorer.transform(queries)
    out = topk[['qid', 'docno', 'score']]
    out.to_csv(os.path.join(args.sink, f'{args.scorer}.{args.topk}.tsv'), sep='\t', index=False, header=False)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)