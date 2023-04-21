import fire 
import os 
import subprocess as sp
import logging
from os.path import join
from tqdm.auto import tqdm

targets = ['t5', 'tasb', 'colbert', 'bm25', 'electra']
detectors = ['regression', 'bert']
alpha = [0., 0.1, 0.2, 0.3, 0.4, 0.5]
mode = ['context', 'static']
types = ['posiiton', 'salience']
salience = ['sentence', 't5']

def main(script_path : str, 
         candidate_propa : str,
         candidate_rel : str,
         topk_p : str, 
         rank_store : str, 
         outpath : str,):
    njobs = len(targets) * len(detectors) * len(alpha) * len(mode) * 3
    pbar = tqdm(total=njobs)
    for target in targets:
        for detector in detectors:
            topk_propa = join(topk_p, f'{detector}.{target}.1000.tsv')
            topk_rel = join(rank_store, f'{target}.1000.tsv')
            filtered = join(rank_store, f'{target}.10.tsv')
            for a in alpha:
                for t in types:
                    if t == 'salience':
                        for s in salience:
                            for m in mode:
                                inj_propa = join(candidate_propa, f'{m}.{s}.{detector}.tsv')
                                inj_rel = join(candidate_rel, f'{target}.{m}.{s}.csv')
                                out = join(outpath, f'{target}.{detector}.{m}.{s}.{a}.tsv')
                                sp.run(['python', script_path, 
                                        '--injectionpath', inj_propa,
                                        '--rankpath', topk_propa,
                                        '--injectionscorespath', inj_rel,
                                        '--rankscorespath', topk_rel,
                                        '--rankfilterpath', filtered, 
                                        '--outpath', out,
                                        '--alpha', str(a),
                                        '--salient',
                                        '--retriever', target, 
                                        '--detector', detector])
                                pbar.update(1)
                    else:
                        for m in mode:
                            inj_propa = join(candidate_propa, f'{m}.position.{detector}.tsv')
                            inj_rel = join(candidate_rel, f'{target}.{m}.position.csv')
                            out = join(outpath, f'{target}.{detector}.{m}.position.{a}.tsv')
                            sp.run(['python', script_path, 
                                    '--injectionpath', inj_propa,
                                    '--rankpath', topk_propa,
                                    '--injectionscorespath', inj_rel,
                                    '--rankscorespath', topk_rel,
                                    '--rankfilterpath', filtered, 
                                    '--outpath', out,
                                    '--alpha', str(a),
                                    '--retriever', target, 
                                    '--detector', detector])
                            pbar.update(1)
                        

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    fire.Fire(main)