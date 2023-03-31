import logging
import fire 
import util
from bert import train_bert, test_bert
from logistic import train_regression, test_regression
funcs = {
    'bert' : (train_bert, test_bert),
    'regression' : (train_regression, test_regression)
}

def main(model : str, 
         dataset_path : str, 
         out_dir : str,
         epochs : int = None, 
         batch_size : int = 1
         ):

    if util.init_out(out_dir) == 1: 
        logging.error('Check path formatting at: {out_dir}, this is a file...')
        exit
    train_func, test_func = funcs[model]
    dataset = util.load_dataset(dataset_path, model)

    model_params = {
        'epochs' : epochs,
        'batch_size' : batch_size,
        'out_dir' : out_dir
    }

    result = train_func(dataset, **model_params)
    eval = 
    return 0



if __name__=='__main__':
    fire.Fire(main)