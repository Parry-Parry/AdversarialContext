import logging
import fire
import subprocess as sp
import os 
from tqdm.auto import tqdm

models = ['bm25', 'tasb', 'electra', 't5']
types = ['position'] # salience
salience = ['sentence', 't5']
injections = ['static', 'context']

special_args = {
    'bm25' : ['--dataset', 'msmarco_passage'],
    't5' : [],
    'electra' : [],
    'tasb' : ['--checkpoint', 'sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco']
}

def main(script_name : str, inject_store : str, rank_store : str, out_dir : str):
    njobs = len(models) * len(injections) * 3
    pbar = tqdm(total=njobs)
    main_args = ['python', script_name, '-full_path', rank_store, '-qrels', 'msmarco-passage/trec-dl-2019/judged']
    for injection in injections:
        for type in types:
            if type=='salience':
                for sal in salience:
                    for model in models:
                        args = main_args
                        args.extend(['-source', os.path.join(inject_store, f'{injection}.{sal}.tsv')])
                        args.extend(['-scorer', model])
                        args.extend(['-type', type])
                        args.extend(['-sink', os.path.join(out_dir, f'{model}.{injection}.{sal}.csv')])
                        args.extend(special_args[model])
                        logging.info(f'Now running: {" ".join(args)}')
                        sp.run(args)
                        pbar.update(1)
            else:
                for model in models:
                    args = main_args
                    args.extend(['-source', os.path.join(inject_store, f'{injection}.{type}.tsv')])
                    args.extend(['-scorer', model])
                    args.extend(['-type', type])
                    args.extend(['-sink', os.path.join(out_dir, f'{model}.{injection}.{type}.csv')])
                    args.extend(special_args[model])
                    logging.info(f'Now running: {" ".join(args)}')
                    sp.run(args)
                    pbar.update(1)

if __name__=='__main__':
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
    fire.Fire(main)