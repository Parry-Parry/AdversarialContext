import pyterrier as pt
pt.init()
import argparse
import os
import ir_datasets
import pandas as pd
from pyterrier_t5 import MonoT5ReRanker

_logger = ir_datasets.log.easy()

parser = argparse.ArgumentParser()

parser.add_argument('-qrels', type=str)
parser.add_argument('-top', type=int)
parser.add_argument('-sink', type=str)

def build_data(path):
  result = []
  dataset = ir_datasets.load(f'msmarco-passage/{path}')
  docs = dataset.docs_store()
  queries = {q.query_id: q.text for q in dataset.queries_iter()}
  for qrel in _logger.pbar(ir_datasets.load(f'msmarco-passage/{path}').scoreddocs_iter(), desc='dev data'):
    if qrel.query_id in queries:
      result.append([qrel.query_id, queries[qrel.query_id], qrel.doc_id, docs.get(qrel.doc_id).text])
  return pd.DataFrame(result, columns=['qid', 'query', 'docno', 'text'])

def main(args):
    dataset = pt.get_dataset("irds:msmarco-passage")
    bm25 = pt.BatchRetrieve.from_dataset('msmarco_passage', 'terrier_stemmed', wmodel='BM25')

    monoT5 = MonoT5ReRanker()
    scorer = bm25 >> pt.text.get_text(dataset, "text") >> monoT5 % args.top

    data = build_data(args.qrels)

    queries = data[['qid', 'query']].copy().drop_duplicates()

    topk = scorer.transform(queries)
    out = topk[['qid', 'docno', 'score']]
    out.to_csv(os.path.join(args.sink, 'T5.tsv'), sep='\t', index=False, header=False)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)