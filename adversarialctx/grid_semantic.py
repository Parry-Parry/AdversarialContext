import logging
import fire
import subprocess as sp
import os 
from tqdm.auto import tqdm

suffixes = ['t5', 'sentence', 'position']

def main(script_name : str, inject_store : str, rank_store : str, out_dir : str):
    njobs = len(suffixes)
    pbar = tqdm(total=njobs)
    for suffix in suffixes:
        args = ['python', script_name]
        args.extend(
            [
            '-ctxsource',
            os.path.join(inject_store, f'context.{suffix}.tsv'),
            '-staticsource',
            os.path.join(inject_store, f'static.{suffix}.tsv'),
            '-full_path',
            rank_store,
            '-embedding_model',
            'nq-distilbert-base-v1',
            '-qrels',
            'msmarco-passage/trec-dl-2019/judged',
            '-sink',
            os.path.join(out_dir, f'semantic.{suffix}.csv')
            ]
        )
        sp.run(args)
        pbar.update(1)
if __name__=='__main__':
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
    fire.Fire(main)