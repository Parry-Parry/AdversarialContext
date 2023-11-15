import os
from os.path import exists, isdir, join

import numpy as np

def init_out(path : str):
    if exists(path):
        if isdir(path):
            if not exists(join(path, 'logs')):
                os.mkdir(join(path, 'logs')) 
                os.mkdir(join(path, 'models')) 
            return 0
        else: return 1
    else:
        os.mkdir(path)
        os.mkdir(join(path, 'logs')) 
        os.mkdir(join(path, 'models')) 
        return 0

def load_dataset(path : str):
    with open(path, 'r') as f:
        items = map(lambda x : x.strip().split('\t'), f.readlines())
    
    x, y = map(list, zip(*items))
    return x, np.array(y, dtype=np.int8)