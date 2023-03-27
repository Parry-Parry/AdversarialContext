import logging
import fire
import subprocess as sp
import os 
from tqdm.auto import tqdm

scripts = ['AdversarialContext/adversarialctx/injection/syringes/salient/inject_sentences.py', 'AdversarialContext/adversarialctx/injection/syringes/position/inject_sentences.py', 'AdversarialContext/adversarialctx/injection/syringes/salient/inject_context.py', 'AdversarialContext/adversarialctx/injection/syringes/position/inject_context.py']

def main():
    njobs = 4
    pbar = tqdm(total=njobs)
    args = [
        'python'
    ]
    for script in scripts:
        tmp = args
        tmp.append(script)
        if 'sentences' in script: tmp.extend(['-sentence_source', 'context/static.tsv'])
        else: tmp.extend(['-sentence_source', 'context/generated.tsv'])
        tmp.extend(['-source', 'context/pairs.tsv', '-dataset', 'trec-dl-2019/judged-sink'])
        if 'salient' in script: tmp.extend(['-sink', 'context/injection'])
        else:
            if 'context' in script: tmp.extend(['-sink', 'context/injection/context.position.tsv'])
            else: tmp.extend(['-sink', 'context/injection/static.position.tsv'])
        sp.run(tmp)
        pbar.update(1)


if __name__=='__main__':
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
    fire.Fire(main)