import os
from os.path import exists, isdir, join

def init_out(path : str):
    if exists(path):
        if isdir(path):
            if not exists(join(path, 'logs')):
                os.mkdir(join(path, 'logs')) 
                os.mkdir(join(path, 'models')) 
            return 0
        else: return 0
    else:
        os.mkdir(path)
        os.mkdir(join(path, 'logs')) 
        os.mkdir(join(path, 'models')) 
        return 0

def load_dataset(path : str, model : str):
    if model == 'bert': pass 
    if model == 'regression' : pass 
    return None

def create_iterator(dataset):
    pass