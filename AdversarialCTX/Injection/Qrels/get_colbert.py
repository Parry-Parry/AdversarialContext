import pyterrier as pt
pt.init()
import argparse
import os
from pyterrier_colbert.ranking import ColBERTFactory
import ir_datasets
import pandas as pd

_logger = ir_datasets.log.easy()

parser = argparse.ArgumentParser()

parser.add_argument('-qrels', type=str)
parser.add_argument('-top', type=int)
parser.add_argument('-sink', type=str)
parser.add_argument('-chkpt', type=str)
parser.add_argument('-idx_path', type=str)
parser.add_argument('-idx', type=str)

def build_data(path):
  result = []
  dataset = ir_datasets.load(f'msmarco-passage/{path}')
  docs = dataset.docs_store()
  queries = {q.query_id: q.text for q in dataset.queries_iter()}
  for qrel in _logger.pbar(ir_datasets.load('msmarco-passage/dev').scoreddocs, desc='dev data'):
    if qrel.query_id in queries:
      result.append([qrel.query_id, queries[qrel.query_id], qrel.doc_id, docs.get(qrel.doc_id).text])
  return pd.DataFrame(result, columns=['qid', 'query', 'docno', 'text'])

def main(args):
    data = build_data(args.qrels)
    queries = data[['qid', 'query']].copy().drop_duplicates()

    pytcolbert = ColBERTFactory(args.chkpt, args.idx_path, args.idx)
    scorer = pytcolbert.end_to_end() % args.top

    topk = scorer.transform(queries)
    out = topk[['qid', 'docno', 'score']]
    out.to_csv(os.path.join(args.sink, 'COLBERT.tsv'), sep='\t', index=False, header=False)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)