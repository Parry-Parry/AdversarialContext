from parryutil import yaml_load
from . import ABNIRML, build_rank_lookup, MRC
from fire import Fire
import pandas as pd
import pyterrier as pt 
if not pt.started():
    pt.init()
from pyterrier.io import read_results

def pairwise(config : str): 
    config = yaml_load(config)
    original_file = config['original_file']
    adversarial_file = config['adversarial_file']
    out_file = config['out_file']

    original = read_results(original_file)
    adversarial = read_results(adversarial_file)

    original = original.sort_values(by=['qid', 'score'], ascending=[True, False])
    adversarial = adversarial.sort_values(by=['qid', 'score'], ascending=[True, False])

    original_lookup = build_rank_lookup(original)

    original['original_score'] = original['score']
    original.drop(columns=['score'], inplace=True)

    adversarial = adversarial.merge(original, on=['qid', 'docno'], how='left')
    adversarial['ABNIRML'] = adversarial.apply(lambda x : ABNIRML(x['score'], x['original_score']), axis=1)
    adversarial['MRR'] = adversarial.apply(lambda x : MRC(x['docno'], x['score'], original_lookup[x['qid']]), axis=1)

    adversarial.to_csv(out_file, sep='\t', index=False)

    return "Done!"

if __name__ == '__main__':
    Fire(pairwise)
