from fire import Fire
from parryutil import load_yaml
from typing import Union
import os
from typing import Union

from contextgen.evaluation.retrieval import retrieval_score

def main(config : Union[str, dict], dir : str = None, out_dir : str = None):
    main_config = load_yaml(config) if isinstance(config, str) else config

    dir = main_config.pop('dir') if not dir else dir
    out_dir = main_config.pop('out_dir') if not out_dir else out_dir
    qrels = main_config.pop('qrels')

    for _, config in main_config.items():
        original_file = config['original_file']
        injection_file = config['injection_file']
        out_file = os.path.join(out_dir, injection_file)
        print(f"Running {injection_file}")
        retrieval_score(original_file, injection_file, out_file, qrels)
    return "Done!"

if __name__ == "__main__":
    Fire(main)