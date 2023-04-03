import logging
import os
import fire
import torch 
import util
from bert import train_bert, test_bert
from logistic import train_regression, test_regression

funcs = {
    'bert' : (train_bert, test_bert),
    'regression' : (train_regression, test_regression)
}

def main(model_name : str, 
         dataset_path : str, 
         out_dir : str,
         n_class : int = 2,
         epochs : int = 1, 
         batch_size : int = 1,
         model_id : str = None,
         ncpu : int = 1
         ):

    if util.init_out(out_dir) == 1: 
        logging.error('Check path formatting at: {out_dir}, this is a file...')
        exit
    train_func, test_func = funcs[model_name]
    train = util.load_dataset(os.path.join(dataset_path, 'train.tsv'))
    test = util.load_dataset(os.path.join(dataset_path, 'test.tsv'))

    model_params = {
        'model_id' : model_id,
        'n_class' : n_class,
        'epochs' : epochs,
        'batch_size' : batch_size,
        'ncpu' : ncpu,
        'out_dir' : os.path.join(out_dir, 'logs'),
        'out' : os.path.join(out_dir, 'models')
    }

    result = train_func(train, **model_params)
    model = result[0]
    if model_name == 'regression':
        model_params['encoder'] = result[1]
    eval = test_func(test, model, **model_params)

    print(eval)
    with open(os.path.join(model_params['out_dir'], f'{model_name}.{epochs}.eval.tsv'), 'w') as f:
        f.write(f'{eval["accuracy"]}\t{eval["f1"]}\t{eval["precision"]}\t{eval["recall"]}\n')
    
    return f"Training of {model_name} Completed!"


if __name__=='__main__':
    fire.Fire(main)