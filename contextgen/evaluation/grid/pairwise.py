from fire import Fire
from contextgen import load_yaml
from typing import Union
import os
from contextgen.evaluation.pairwise import pairwise_score

def main(config : Union[str, dict], dir : str = None, out_dir : str = None):
    main_config = load_yaml(config) if isinstance(config, str) else config

    dir = main_config.pop('dir') if not dir else dir
    out_dir = main_config.pop('out_dir') if not out_dir else out_dir
    original_dir = main_config.pop('original_dir')
    
    for _, config in main_config.items():
        original_file = config['original_file']
        original_file = os.path.join(original_dir, original_file)
        model = config['model']
        files = [f for f in os.listdir(dir) if model in f]
        for injection_file in files:
            out_file = os.path.join(out_dir, injection_file)
            injection_file = os.path.join(dir, injection_file)
            if os.path.exists(out_file): 
                print(f"Already done {injection_file}")
                continue
            print(f"Running {injection_file}")
            pairwise_score(original_file, injection_file, out_file)
    return "Done!"

if __name__ == "__main__":
    Fire(main)