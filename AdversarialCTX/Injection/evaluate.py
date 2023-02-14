import argparse
from collections import defaultdict
import numpy as np
import os
import pandas as pd
import ir_datasets

_logger = ir_datasets.log.easy()
    
def build_data(path):
  result = []
  dataset = ir_datasets.load(f'msmarco-passage/{path}')
  docs = dataset.docs_store()
  queries = {q.query_id: q.text for q in dataset.queries_iter()}
  for qrel in _logger.pbar(ir_datasets.load('msmarco-passage/dev').scoreddocs, desc='dev data'):
    if qrel.query_id in queries:
      result.append([qrel.query_id, queries[qrel.query_id], qrel.doc_id, docs.get(qrel.doc_id).text])
  return pd.DataFrame(result, columns=['qid', 'query', 'docno', 'text'])


parser = argparse.ArgumentParser()

parser.add_argument('-source', type=str)
parser.add_argument('-qrels', type=str)
parser.add_argument('-sink', type=str)

def main(args):
    ds = ir_datasets.load("irds:msmarco-passage")
    queries = pd.DataFrame(ds.queries_iter()).set_index('query_id').text.to_dict()

    def build_from_df(df):
        new = []
        for row in texts.itertuples():
            new.append({'qid':row.qid, 'query':queries[row.qid], 'docno':row.docno, 'text':row.adversary})
        return pd.DataFrame.from_records(new)

    cols = ['qid', 'docno', 'score', 'adversary']
    types = {'qid' : str, 'docno' : str, 'score' : float, 'adversary' : str}

    texts = pd.read_csv(args.source, sep='\t', header=None, index_col=False, names=cols, dtype=types)

    new_texts = build_from_df(texts)
