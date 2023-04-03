import fire 
import os
import pandas as pd

def main(dir : str, out : str):
    files = [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]

    frame = []
    cols = ['qid', 'docno', 'context', 'pos', 'salience', 'orginal_score', 'new_score']
    types = {c : str for c in cols if 'score' not in c}
    types['original_score'] = float
    types['new_score'] = float

    for file in files:
        parts = file.split('.')
        tmp = pd.read_csv(os.path.join(dir, file), names=cols, dtype=types)
        tmp['model'] = parts[0]
        tmp['injection'] = parts[1]
        tmp['type'] = parts[2]
        tmp['eval'] = parts[3]
        frame.append(tmp)
    
    pd.concat(frame).to_csv(out, index=False)

if __name__=='__main__':
    fire.Fire(main)
    

