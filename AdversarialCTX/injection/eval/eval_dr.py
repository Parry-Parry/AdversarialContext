import pyterrier as pt
pt.init()
import argparse
import os
import pandas as pd
import ir_datasets
from pyterrier_t5 import ElectraScorer
from pyterrier_dr import ElectraScorer, TasB
import logging

scorers = {
    'electra' : ElectraScorer,
    'tasb' : TasB
}

parser = argparse.ArgumentParser()

parser.add_argument('-source', type=str)
parser.add_argument('-scorer', type=str)
parser.add_argument('-qrels', type=str)
parser.add_argument('-sink', type=str)

parser.add_argument('--model_name', type=str)

def main(args):
    ds = ir_datasets.load(f"msmarco-passage/{args.qrels}")
    queries = pd.DataFrame(ds.queries_iter()).set_index('query_id').text.to_dict()

    if args.scorer == 'tasb': assert args.model_name is not None
    try:
        if args.model_name:  model = scorers[args.scorer](args.model_name)
        else: model = scorers[args.scorer]()
    except KeyError:
        logging.error(f'Model: {args.scorer} not found')
        exit
    
    scorer = model

    def build_from_df(df):
        new = []
        for row in df.itertuples():
            new.append({'qid':row.qid, 'query':queries[row.qid], 'docno':row.docno, 'text':row.adversary})
        return pd.DataFrame.from_records(new)

    cols = ['qid', 'docno', 'score', 'adversary', 'rel', 'pos']
    types = {'qid' : str, 'docno' : str, 'score' : float, 'adversary' : str, 'rel' : int, 'pos':str}

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



