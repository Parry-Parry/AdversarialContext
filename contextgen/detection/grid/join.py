from fire import Fire 
import os 
from ..join_scores import interpolated_scores
import numpy as np

def grid_join(score_dir : str, rel_dir : str, out_dir : str, max_alpha : float):
    for alpha in np.arange(0, max_alpha + 0.1, 0.1):
        alpha_name = str(alpha).replace('.', '_')
        out = os.path.join(out_dir, alpha_name)
        os.makedirs(out, exist_ok=True)
        model = rel_dir.split('/')[-1]
        
        for file in os.listdir(score_dir):
            score_file = os.path.join(score_dir, file)
            rel_file = '.'.join([model, *file.split('.')[1:]])
            rel_file = os.path.join(rel_dir, rel_file)
            out_file = os.path.join(out, file)
            interpolated_scores(rel_file, score_file, out_file, alpha)
    
    return "Done!"

if __name__ == '__main__':
    Fire(grid_join)