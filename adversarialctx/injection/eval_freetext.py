from collections import defaultdict
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

def build_rank_lookup(df):
    frame = defaultdict(dict)
    for qid in df.qid.unique().tolist():
        sub = df[df.qid==qid].sort_values(by='score').reset_index
        for row in sub.itertuples():
            frame[qid][row.docno] = row.index
    

scorers = {
    'tasb' : init_dr,
    'electra' : init_dr,
    't5' : init_monot5,
    'colbert' : init_colbert,
    'bm25' : init_bm25
}

parser = argparse.ArgumentParser()

parser.add_argument('-source', type=str)
parser.add_argument('-sentence', type=str)
parser.add_argument('-scorer', type=str)
parser.add_argument('-qrels', type=str)
parser.add_argument('-sink', type=str)

parser.add_argument('--dataset', type=str, default=None)
parser.add_argument('--checkpoint', type=str, default=None)
parser.add_argument('--gpu', action='store_true')

def main(args):
    with open(args.sentence, 'r') as f:
       sentences = f.readlines()

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

    cols = ['qid', 'docno', 'score', 'adversary', 'rel', 'pos', 'salience', 'salience_type', 'sentence']
    types = {'qid' : str, 'docno' : str, 'score' : float, 'adversary' : str, 'rel' : str, 'pos':str, 'salience':str, 'salience_type':str, 'sentence':str}

    advers = [f for f in os.listdir(args.source) if os.path.isfile(os.path.join(args.source, f))]
    frames = []
    for text in advers:
        texts = pd.read_csv(os.path.join(args.source, text), dtype=types)
        for sal in ['salient', 'nonsalient']:
            subset = texts[texts.salience==sal]

            test = build_from_df(subset)
            test['query'] = test['query'].apply(preprocess)
            test['text'] = test['text'].apply(preprocess)
            results = scorer(test)

            old_lookup = build_rank_lookup(subset)
            new_lookup = build_rank_lookup(results)

            def get_rank_change(qid, docno):
                rank_change = old_lookup[qid][docno] - new_lookup[qid][docno]
                return rank_change

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
            
            for sentence in sentences:
                subset = results[results.sentence == sentence]
                frames = []
                for qid in queries.keys():
                    subset = subset[subset.qid == qid]
                    comparison = texts[texts.qid==qid]

                    frames.append(comparison, subset)


                subset['adv_signal'] = texts.apply(lambda x : ABNIRML(x['qid'], x['docno'], x['score']), axis=1)
                subset['rank_change'] = texts.apply(lambda x : get_rank_change(x['qid'], x['docno'], x['score']), axis=1)
                subset['adv_score'] = texts.apply(lambda x : get_score(x['qid'], x['docno']), axis=1)
                frames.append(subset)
                
    pd.concat(frames).to_csv(os.path.join(args.sink, f'abnirml.csv'))

if __name__ == '__main__':
    args = parser.parse_args()
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
    main(args)