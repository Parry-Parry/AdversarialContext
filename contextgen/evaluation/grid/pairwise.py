from fire import Fire
from parryutil import load_yaml
from typing import Union
import os
from os import listdir
from contextgen.evaluation.pairwise import pairwise_score

def main(config : Union[str, dict], dir : str = None, out_dir : str = None):
    config = load_yaml(config) if isinstance(config, str) else config

    dir = config['dir'] if not dir else dir
    out_dir = config['out_dir'] if not out_dir else out_dir
    model_config = config['model_config']
    original_file = config['original_file']
    files = listdir(dir)

    for file in files:
        injection_file = os.path.join(dir, file)
        out_file = os.path.join(out_dir, file)
        print(f"Running {injection_file}")
        pairwise_score(original_file, injection_file, out_file, model_config)
    
    return "Done!"

if __name__ == "__main__":
    Fire(main)