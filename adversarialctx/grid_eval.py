import logging
import fire
import subprocess as sp
import os 
from tqdm.auto import tqdm

models = ['bm25', 'tasb', 'electra', 't5']
model = ['colbert']
types = ['salience', 'position']
salience = ['sentence', 't5']
injections = ['static', 'context'] # Static Complete

special_args = {
    'bm25' : ['--dataset', 'msmarco_passage'],
    't5' : [],
    'electra' : [],
    'tasb' : ['--checkpoint', 'sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco'],
    'colbert' : ['--checkpoint', 'http://www.dcs.gla.ac.uk/~craigm/ecir2021-tutorial/colbert_model_checkpoint.zip']
}

def main(script_name : str, inject_store : str, rank_store : str, out_dir : str, colbert : bool = False):
    njobs = len(models) * len(injections) * 3
    pbar = tqdm(total=njobs)
    if colbert: models = model
    for injection in injections:
        for type in types:
            if type=='salience':
                for sal in salience:
                    for model in models:
                        args = ['python', script_name, '-full_path', rank_store, '-qrels', 'msmarco-passage/trec-dl-2019/judged']
                        args.extend(['-source', os.path.join(inject_store, f'{injection}.{sal}.tsv')])
                        args.extend(['-scorer', model])
                        args.extend(['-type', type])
                        args.extend(['-sink', os.path.join(out_dir, f'{model}.{injection}.{sal}.csv')])
                        args.extend(special_args[model])
                        if injection == 'context': args.append('--context')
                        logging.info(f'Now running: {" ".join(args)}')
                        sp.run(args)
                        pbar.update(1)
            else:
                for model in models:
                    args = ['python', script_name, '-full_path', rank_store, '-qrels', 'msmarco-passage/trec-dl-2019/judged']
                    args.extend(['-source', os.path.join(inject_store, f'{injection}.{type}.tsv')])
                    args.extend(['-scorer', model])
                    args.extend(['-type', type])
                    args.extend(['-sink', os.path.join(out_dir, f'{model}.{injection}.{type}.csv')])
                    args.extend(special_args[model])
                    if injection == 'context': args.append('--context')
                    logging.info(f'Now running: {" ".join(args)}')
                    sp.run(args)
                    pbar.update(1)

if __name__=='__main__':
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
    fire.Fire(main)