import logging
import fire
import subprocess as sp
import os 
from tqdm.auto import tqdm


types = ['salience', 'position']
salience = ['sentence', 't5']
injections = ['static', 'context'] 


def main(script_name : str, model_path : str, inject_store : str, rank_store : str, out_dir : str):
    models = ['bm25', 'tasb', 'electra', 't5', 'colbert']
    propmodels = ['regression']
    njobs = len(models) * len(injections) * 3 * len(propmodels)
    pbar = tqdm(total=njobs)
    for injection in injections:
        for type in types:
            if type=='salience':
                for sal in salience:
                    for model in models:
                        for pmodel in propmodels:
                            args = ['python', script_name, '--modelpath', model_path, '--originalpath', os.path.join(rank_store, f'{model}.10.tsv'), '--dataset', 'msmarco-passage']
                            args.extend(['--advpath', os.path.join(inject_store, f'{injection}.{sal}.tsv')])
                            args.extend(['--modeltype', pmodel])
                            args.extend(['--type', type])
                            args.extend(['--out', os.path.join(out_dir, f'{model}.{injection}.{sal}.{pmodel}.csv')])
                            if injection == 'context': args.append('--context')
                            logging.info(f'Now running: {" ".join(args)}')
                            sp.run(args)
                            pbar.update(1)
            else:
                for model in models:
                    for pmodel in propmodels:
                        args = ['python', script_name, '--modelpath', model_path, '--originalpath', os.path.join(rank_store, f'{model}.10.tsv'), '--dataset', 'msmarco-passage']
                        args.extend(['--advpath', os.path.join(inject_store, f'{injection}.{type}.tsv')])
                        args.extend(['--modeltype', pmodel])
                        args.extend(['--type', type])
                        args.extend(['--out', os.path.join(out_dir, f'{model}.{injection}.{type}.{pmodel}.csv')])
                        if injection == 'context': args.append('--context')
                        logging.info(f'Now running: {" ".join(args)}')
                        sp.run(args)
                        pbar.update(1)

if __name__=='__main__':
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
    fire.Fire(main)