import pyterrier as pt
pt.init()
import ir_datasets
import pandas as pd
import argparse
import logging
from ..terrier_lexrank import LexRanker

parser = argparse.ArgumentParser()
parser.add_argument('-qrels', type=str)
parser.add_argument('-sink', type=str)

parser.add_argument('--index', type=str, default=None)
parser.add_argument('--verbose', action='store_true')

def main(args):
    ds = ir_datasets.load("msmarco-passage")
    text = pd.DataFrame(ds.docs_iter()).set_index('doc_id').text.to_dict()
    qrels = pd.DataFrame(ir_datasets.load(f"msmarco-passage/{args.qrels}").qrels_iter())
    docs = qrels['doc_id'].unique().tolist()
    
    inp = []
    for doc in docs:
        inp.append({'docno':doc, 'text':text[doc]})
    inp = pd.DataFrame.from_records(inp)
    
    if args.index:
        ds = pt.get_dataset(args.index)
        index = ds.get_index(variant='terrier_stemmed')
    else:
        index=None

    ranker = LexRanker(setting='ranks', background_index=index, norm=True, threshold=.1, verbose=args.verbose)

    logging.info('Computing salient positions...')
    out = ranker.transform(inp)

    logging.info('Saving...')
    out.to_csv(args.sink)

if __name__ == '__main__':
    args = parser.parse_args()
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
    main(args)