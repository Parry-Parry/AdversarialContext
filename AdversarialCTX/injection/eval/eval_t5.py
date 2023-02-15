import pyterrier as pt
pt.init()
import argparse
import os
import pandas as pd
import ir_datasets
from pyterrier_t5 import MonoT5ReRanker
import re

def clean_text(text):
    text = re.sub(r'[^A-Za-z0-9 ]+', '', text)
    return re.sub(r'/[^\x00-\x7F]/g', '', text).strip()

parser = argparse.ArgumentParser()

parser.add_argument('-source', type=str)
parser.add_argument('-qrels', type=str)
parser.add_argument('-sink', type=str)

def main(args):
    ds = ir_datasets.load(f"msmarco-passage/{args.qrels}")
    queries = pd.DataFrame(ds.queries_iter()).set_index('query_id').text.to_dict()

    dataset = pt.get_dataset("irds:msmarco-passage")
    bm25 = pt.BatchRetrieve.from_dataset('msmarco_passage', 'terrier_stemmed_text', wmodel='BM25', metadata=['docno', 'text'])

    monoT5 = MonoT5ReRanker()
    scorer = bm25 >> pt.text.get_text(dataset, "text") >> monoT5 

    def build_from_df(df):
        new = []
        for row in df.itertuples():
            new.append({'qid':row.qid, 'query':queries[row.qid], 'docno':row.docno, 'text':row.adversary})
        return pd.DataFrame.from_records(new)

    cols = ['qid', 'docno', 'score', 'adversary']
    types = {'qid' : str, 'docno' : str, 'score' : float, 'adversary' : str}

    advers = [f for f in os.listdir(args.source) if os.path.isfile(os.path.join(args.source, f))]

    frames = []
    for text in advers:
      texts = pd.read_csv(os.path.join(args.source, text), sep='\t', header=None, index_col=False, names=cols, dtype=types)

      test = build_from_df(texts)
      test['query'] = test['query'].apply(clean_text)
      test['text'] = test['text'].apply(clean_text)
      results = scorer(test)

      def ABNIRML(qid, docno, score):
        tmp = results[results['qid']==qid].set_index('docno')['score']
        adv_score = tmp.loc[docno]
        diff = score - adv_score
        if diff < 0: return -1 
        elif diff > 0: return 1
        return 0

      texts['adv_score'] = texts.apply(lambda x : ABNIRML(x['qid'], x['docno'], x['score']))
      texts['file'] = text
      frames.append(texts)

    out = pd.concat(frames)
    out.to_csv(args.sink)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)



