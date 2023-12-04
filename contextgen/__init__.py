from nltk.tokenize import sent_tokenize
from yaml import load
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader
import json
import logging
import subprocess as sp

def batch_iter(items, batch_size):
    i = 0
    while i < (len(items) // batch_size):
        i += 1
        yield items[i*batch_size:min((i+1)*batch_size, len(items))]

def parse_span(text):
    spans = sent_tokenize(text)
    if len(spans) == 1: return text
    elif len(spans) == 0: return text
    else: return spans[0]

def load_yaml(path : str) -> dict:
    return load(open(path), Loader=Loader)

def execute(config_path : str, default_path : str = None):
    executions = load_yaml(config_path)
    if default_path is not None: defaults = load_yaml(default_path)
    for k, cfg in executions.items():
        if default_path is not None: cfg['args'].update(defaults)  
        logging.info('\n'.join([f'EXECUTION NAME: {k}', 'ARGS:', json.dumps(cfg['args'], indent=2)]))
        cmd = ['python', '-m', cfg['script']]
        for arg, val in cfg['args'].items():
            cmd.append(f'--{arg}')
            if val is not None:
                if type(val) == list:
                    cmd.append(' '.join(val))
                    continue
                cmd.append(str(val))
        sp.run(cmd)
    
    return f'Completed {len(executions)} executions.'