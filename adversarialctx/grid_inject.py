import logging
import fire
import subprocess as sp
import os 
from tqdm.auto import tqdm

scripts = ['AdversarialContext/adversarialctx/injection/syringes/salient/inject_sentences.py', 'AdversarialContext/adversarialctx/injection/syringes/position/inject_sentences.py', 'AdversarialContext/adversarialctx/injection/syringes/salient/inject_context.py', 'AdversarialContext/adversarialctx/injection/syringes/position/inject_context.py']

def main(gen : str = 'context/generated.tsv', out : str = 'context/injection'):
    njobs = len(scripts)
    pbar = tqdm(total=njobs)

    for script in scripts:
        args = ['python']
        args.append(script)
        if 'sentences' in script: args.extend(['-sentence_source', 'context/static.tsv'])
        else: args.extend(['-sentence_source', gen])
        args.extend(['-source', 'context/pairs.tsv', '-dataset', 'trec-dl-2019/judged'])
        if 'salient' in script: args.extend(['-sink', out])
        else:
            if 'context' in script: args.extend(['-sink', os.path.join(out, 'context.position.tsv')])
            else: args.extend(['-sink', os.path.join(out, 'static.position.tsv')])
        sp.run(args)
        pbar.update(1)


if __name__=='__main__':
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
    fire.Fire(main)