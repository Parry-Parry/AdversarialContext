import pyterrier as pt
pt.init()
import argparse
import os
import pandas as pd
import ir_datasets
from pyterrier_colbert.ranking import ColBERTFactory

parser = argparse.ArgumentParser()

parser.add_argument('-source', type=str)
parser.add_argument('-qrels', type=str)
parser.add_argument('-sink', type=str)
parser.add_argument('-chkpt', type=str)
parser.add_argument('-idx_path', type=str)
parser.add_argument('-idx', type=str)

def main(args):
    ds = ir_datasets.load(f"msmarco-passage/{args.qrels}")
    queries = pd.DataFrame(ds.queries_iter()).set_index('query_id').text.to_dict()

    pytcolbert = ColBERTFactory(args.chkpt, args.idx_path, args.idx)
    pytcolbert.faiss_index_on_gpu = False 
    scorer = pytcolbert.end_to_end() 

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
      results = scorer(test)

      def ABNIRML(qid, docno, score):
        tmp = results[results['qid']==qid].set_index('docno')['score']
        adv_score = tmp.loc[docno]
        diff = score - adv_score
        if diff < 0: return -1 
        elif diff > 0: return 1
        return 0

      texts['adv_score'] = texts.apply(lambda x : ABNIRML(x['qid'], x['docno'], x['score']), axis=1)
      texts['file'] = text
      frames.append(texts)

    out = pd.concat(frames)
    out.to_csv(args.sink)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)



