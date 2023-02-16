import pyterrier as pt
pt.init()
import argparse
import os
import ir_datasets
import pandas as pd
import re
from pyterrier_dr import ElectraScorer

_logger = ir_datasets.log.easy()

def clean_text(text):
    text = re.sub(r'[^A-Za-z0-9 ]+', '', text)
    return re.sub(r'/[^\x00-\x7F]/g', '', text).strip()

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
    bm25 = pt.BatchRetrieve.from_dataset('msmarco_passage', 'terrier_stemmed_text', wmodel='BM25', metadata=['docno', 'text'])

    electra = ElectraScorer()
    scorer = bm25 >> pt.text.get_text(dataset, "text") >> electra % args.top

    data = build_data(args.qrels)

    queries = data[['qid', 'query']].copy().drop_duplicates()
    queries['query'] = queries['query'].apply(clean_text)

    topk = scorer.transform(queries)
    out = topk[['qid', 'docno', 'score']]
    out.to_csv(os.path.join(args.sink, 'ELECTRA.tsv'), sep='\t', index=False, header=False)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)