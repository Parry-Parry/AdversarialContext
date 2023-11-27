from fire import Fire
from parryutil import load_yaml
from typing import Union
import os
from os import listdir

from contextgen.evaluation.score import model_score

def main(config : Union[str, dict], file : str = None, dir : str = None, out_dir : str = None):
    config = load_yaml(config) if isinstance(config, str) else config
    dir = config['dir'] if not dir else dir
    out_dir = config['out_dir'] if not out_dir else out_dir
    model_config = config['model_config']
    model_name = model_config['model']
    files = listdir(dir) if file is None else [file]

    for file in files:
        injection_file = os.path.join(dir, file)
        out_file = os.path.join(out_dir, f'{model_name}.{file}')
        print(f"Running {injection_file}")
        model_score(
            {
                'model_config': model_config,
                'out_file': out_file,
                'ir_dataset': config['ir_dataset'],
                'run_file': injection_file
            }
        )
    return "Done!"

if __name__ == "__main__":
    Fire(main)