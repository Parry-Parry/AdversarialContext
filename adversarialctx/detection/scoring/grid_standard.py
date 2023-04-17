import fire 
import os 
import subprocess as sp
from tqdm.auto import tqdm
import logging

def main(script_name : str, model_path : str, inject_store : str, out_dir : str, sentence : bool = False, standard : bool = False):
    files = [f for f in os.listdir(inject_store) if f.endswith('.tsv')]
    
    propmodels = ['regression', 'bert']
    njobs = len(files) * len(propmodels)
    pbar = tqdm(total=njobs)
    
    for f in files:
        for pmodel in propmodels:
            args = ['python', script_name, '--modelpath', model_path, '--dataset', 'msmarco-passage']
            args.extend(['--datasetpath', os.path.join(inject_store, f'{pmodel}.{f}')])
            args.extend(['--modeltype', pmodel])
            args.extend(['--outpath', os.path.join(out_dir, f)])
            args.extend(['--window_size', str(0)])
            if sentence: args.append('--sentence')
            if standard: args.append('--standard')
            logging.info(f'Now running: {" ".join(args)}')
            sp.run(args)
            pbar.update(1)
                    
if __name__=='__main__':
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
    fire.Fire(main)