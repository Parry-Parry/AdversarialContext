from fire import Fire
from parryutil import load_yaml
from typing import Union
import os
from os import listdir

from contextgen.evaluation.score import model_score

def main(config : Union[str, dict], file : str = None, dir : str = None, out_dir : str = None, trec : bool = False):
    main_config = load_yaml(config) if isinstance(config, str) else config
    dir = main_config.pop('dir') if not dir else dir
    out_dir = main_config.pop('out_dir') if not out_dir else out_dir
    ir_dataset = main_config.pop('ir_dataset')
    files = listdir(dir) if file is None else [file]

    for _, model_config in main_config.items():
        for file in files:
            model_name = model_config['model']
            injection_file = os.path.join(dir, file)
            out_file = os.path.join(out_dir, f'{model_name}.{file}')
            print(f"Running {injection_file}")
            model_score(
                {
                    'model_config': model_config,
                    'out_file': out_file,
                    'ir_dataset': ir_dataset,
                    'run_file': injection_file,
                    'trec' : trec
                }
            )
        return "Done!"

if __name__ == "__main__":
    Fire(main)