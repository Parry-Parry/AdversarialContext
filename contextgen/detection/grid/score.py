from fire import Fire 
import os
from ..score import bert_score

def grid_score(injection_dir : str, 
               out_dir : str, 
               window_size : int = 0,
               model_id : str = 'bert-base-uncased',
               batch_size : int = 128, 
               trec : bool = False, 
               ir_dataset : str = 'msmarco-passage/trec-dl-2019/judged'):
    
    for file in os.listdir(injection_dir):
        injection_file = os.path.join(injection_dir, file)
        out_file = os.path.join(out_dir, file)
        bert_score(model_id, injection_file, out_file, window_size, batch_size, trec, ir_dataset)
    
    return "Done!"

if __name__ == '__main__':
    Fire(grid_score)