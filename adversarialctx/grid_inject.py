import logging
import fire
import subprocess as sp
import os 
from tqdm.auto import tqdm

scripts = ['AdversarialContext/adversarialctx/injection/syringes/salient/inject_sentences.py', 'AdversarialContext/adversarialctx/injection/syringes/position/inject_sentences.py', 'AdversarialContext/adversarialctx/injection/syringes/salient/inject_context.py', 'AdversarialContext/adversarialctx/injection/syringes/position/inject_context.py']

def main():
    njobs = len(scripts)
    pbar = tqdm(total=njobs)

    for script in scripts:
        args = ['python']
        args.append(script)
        if 'sentences' in script: args.extend(['-sentence_source', 'context/static.tsv'])
        else: args.extend(['-sentence_source', 'context/generated.tsv'])
        args.extend(['-source', 'context/pairs.tsv', '-dataset', 'trec-dl-2019/judged'])
        if 'salient' in script: args.extend(['-sink', 'context/injection'])
        else:
            if 'context' in script: args.extend(['-sink', 'context/injection/context.position.tsv'])
            else: args.extend(['-sink', 'context/injection/static.position.tsv'])
        sp.run(args)
        pbar.update(1)


if __name__=='__main__':
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
    fire.Fire(main)