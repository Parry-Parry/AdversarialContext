import pyterrier as pt
pt.init()
import argparse
import pyterrier_dr
from pyterrier_dr import TasB
import logging

parser = argparse.ArgumentParser()

parser.add_argument('-dataset', type=str)

parser.add_argument('--index_name', type=str, default='msmarco-passage')

def main(args):
    index = pyterrier_dr.NumpyIndex(f'{args.index_name}.tasb.np')
    

    model = TasB('sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco')

    logging.info(f'Indexing {args.dataset} with tasb...')
    pipeline = model >> index
    pipeline.index(pt.get_dataset(f'irds:{args.dataset}').get_corpus_iter())


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)



