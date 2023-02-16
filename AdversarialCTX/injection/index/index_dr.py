import pyterrier as pt
pt.init()
import argparse
import pyterrier_dr
from pyterrier_dr import ElectraScorer, TasB
import logging

scorers = {
    'electra' : ElectraScorer,
    'tasb' : TasB
}

parser = argparse.ArgumentParser()

parser.add_argument('-dataset', type=str)
parser.add_argument('-scorer', type=str)

parser.add_argument('--index_name', type=str, default='msmarco-passage')

def main(args):
    index = pyterrier_dr.NumpyIndex(f'{args.index_name}.{args.scorer}.np')
    
    try:
        model = scorers[args.scorer]
    except KeyError:
        logging.error(f'Model: {args.scorer} not found')
        exit

    logging.info(f'Indexing {args.dataset} with {args.scorer}...')
    pipeline = model >> index
    pipeline.index(pt.get_dataset(f'irds:{args.dataset}').get_corpus_iter())


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)



