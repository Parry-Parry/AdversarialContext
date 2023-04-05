import logging
import fire
import subprocess as sp
import os 
from tqdm.auto import tqdm

'''
types = ['salience', 'position']
salience = ['sentence', 't5']
injections = ['static', 'context'] 
'''

types = ['salience']
salience = ['sentence', 't5']
injections = ['context'] 

def main(script_name : str, model_path : str, inject_store : str, rank_store : str, out_dir : str, window_size : int = 0):
    propmodels = ['regression', 'bert']
    njobs = len(injections) * 3 * len(propmodels)
    pbar = tqdm(total=njobs)
    for injection in injections:
        for type in types:
            if type=='salience':
                for sal in salience:
                    for pmodel in propmodels:
                        args = ['python', script_name, '--modelpath', model_path, '--originalpath', rank_store, '--dataset', 'msmarco-passage']
                        args.extend(['--advpath', os.path.join(inject_store, f'{injection}.{sal}.tsv')])
                        args.extend(['--modeltype', pmodel])
                        args.extend(['--type', type])
                        args.extend(['--out', os.path.join(out_dir, f'{injection}.{sal}.{pmodel}.csv')])
                        args.extend(['--window_size', str(window_size)])
                        if injection == 'context': args.append('--context')
                        logging.info(f'Now running: {" ".join(args)}')
                        sp.run(args)
                        pbar.update(1)
            else:
                for pmodel in propmodels:
                    args = ['python', script_name, '--modelpath', model_path, '--originalpath', rank_store, '--dataset', 'msmarco-passage']
                    args.extend(['--advpath', os.path.join(inject_store, f'{injection}.{type}.tsv')])
                    args.extend(['--modeltype', pmodel])
                    args.extend(['--type', type])
                    args.extend(['--out', os.path.join(out_dir, f'{injection}.{type}.{pmodel}.csv')])
                    args.extend(['--window_size', str(window_size)])
                    if injection == 'context': args.append('--context')
                    logging.info(f'Now running: {" ".join(args)}')
                    sp.run(args)
                    pbar.update(1)

if __name__=='__main__':
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
    fire.Fire(main)