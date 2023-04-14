import fire 
import os
import pandas as pd

def main(dir : str, out : str):
    files = [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]

    frame = []
    cols = ['qid', 'docno', 'context', 'pos', 'salience', 'orginal_score', 'new_score']
    types = {}
    types['orginal_score'] = float
    types['new_score'] = float
    types.update({c : str for c in cols if 'score' not in c})
    

    for file in files:
        parts = file.split('.')
        tmp = pd.read_csv(os.path.join(dir, file))
        tmp['injection'] = parts[0]
        tmp['type'] = parts[1]
        tmp['eval'] = parts[2]
        frame.append(tmp)

    frame = pd.concat(frame)

    

    frame.to_csv(out, index=False)

if __name__=='__main__':
    fire.Fire(main)
    

