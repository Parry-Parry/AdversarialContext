import argparse
import logging
import ir_datasets
import pickle
import bz2
from ..terrier_lexrank import split_into_sentences
from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument('-dataset', type=str)
parser.add_argument('-sink', type=str)

def main(args):
    logging.info(f'Loading {args.dataset}...')
    ds = ir_datasets.load(args.dataset)
    docs = list(ds.docs_iter())
    logging.info(f'Now splitting {args.dataset}')
    split_docs = []
    with tqdm(total=len(docs)) as pbar:
        for doc in tqdm(docs):
            split_docs.append(split_into_sentences(doc))
            pbar.update(1)

    with bz2.BZ2File(args.sink, 'wb') as f:
        pickle.dump(split_docs, f)

if __name__ == '__main__':
    args = parser.parse_args()
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
    main(args)