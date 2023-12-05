from fire import Fire
from contextgen import load_yaml
from typing import Union
import os
from typing import Union

from contextgen.evaluation.retrieval import retrieval_score

def main(config : Union[str, dict], dir : str = None, out_dir : str = None):
    main_config = load_yaml(config) if isinstance(config, str) else config

    dir = main_config.pop('dir') if not dir else dir
    out_dir = main_config.pop('out_dir') if not out_dir else out_dir
    qrels = main_config.pop('qrels')
    
    dir = main_config.pop('dir') if not dir else dir
    out_dir = main_config.pop('out_dir') if not out_dir else out_dir
    original_dir = main_config.pop('original_dir')

    all_dirs = os.listdir(dir)
    for _dir in all_dirs:
        for config in main_config.values():
            original_file = config['original_file']
            original_file = os.path.join(original_dir, original_file)
            model = config['model']
            files = [f for f in os.listdir(_dir) if model in f]
            for injection_file in files:
                out_file = os.path.join(out_dir, injection_file)
                injection_file = os.path.join(dir, _dir, injection_file)
                print(f"Running {injection_file}")
                retrieval_score(original_file, injection_file, out_file, qrels)

    return "Done!"

if __name__ == "__main__":
    Fire(main)