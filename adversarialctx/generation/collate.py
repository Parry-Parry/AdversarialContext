import fire
import os

def main(file_path : str, out_path : str, filter : str = None):

    files = [f for f in os.listdir(file_path) if os.path.isfile(os.path.join(file_path, f))]
    if filter: files = [f for f in files if filter in f]

    qidx, didx, indicator = [], [], []

    for file in files:
        with open(os.path.join(file_path, file), 'r') as f:
            text_items = map(lambda x : x.split('\t'), f.readlines())
        q, d, s = map(list, zip(*text_items))

        for item in zip(q, d, s):
            if (q, d) not in indicator: # small size so this search isnt a nightmare
                indicator.append((q, d))
                qidx.append(q)
                didx.append(d)

    with open(out_path, 'w') as f:
        for item in zip(qidx, didx):
            f.write(f'{item[0]}\t{item[1]}\n')

    return 0
    
if __name__=='__main__':
    fire.Fire(main)