import fire 
import os 
import subprocess as sp
from tqdm.auto import tqdm
import logging

types = ['salience', 'position']
salience = ['sentence', 't5']
injections = ['context', 'static'] 

def main(script_name : str, model_path : str, inject_store : str, out_dir : str, sentence : bool = False, standard : bool = False):
    propmodels = ['regression', 'bert']
    njobs = len(injections) * 3 * len(propmodels)
    pbar = tqdm(total=njobs)
    for injection in injections:
        for type in types:
            if type=='salience':
                for sal in salience:
                    for pmodel in propmodels:
                        args = ['python', script_name, '--modelpath', model_path, '--dataset', 'msmarco-passage']
                        args.extend(['--datasetpath', os.path.join(inject_store, f'{injection}.{sal}.tsv')])
                        args.extend(['--modeltype', pmodel])
                        args.extend(['--type', type])
                        args.extend(['--outpath', os.path.join(out_dir, f'{injection}.{sal}.{pmodel}.tsv')])
                        args.extend(['--window_size', str(0)])
                        if sentence: args.append('--sentence')
                        if standard: args.append('--standard')
                        if injection == 'context': args.append('--context')
                        logging.info(f'Now running: {" ".join(args)}')
                        sp.run(args)
                        pbar.update(1)
            else:
                for pmodel in propmodels:
                    args = ['python', script_name, '--modelpath', model_path, '--dataset', 'msmarco-passage']
                    args.extend(['--datasetpath', os.path.join(inject_store, f'{injection}.{type}.tsv')])
                    args.extend(['--modeltype', pmodel])
                    args.extend(['--type', type])
                    args.extend(['--outpath', os.path.join(out_dir, f'{injection}.position.{pmodel}.tsv')])
                    args.extend(['--window_size', str(0)])
                    if injection == 'context': args.append('--context')
                    if sentence: args.append('--sentence')
                    if standard: args.append('--standard')
                    logging.info(f'Now running: {" ".join(args)}')
                    sp.run(args)
                    pbar.update(1)

if __name__=='__main__':
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
    fire.Fire(main)