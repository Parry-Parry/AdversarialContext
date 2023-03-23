import fire
import os

def main(file_path : str, out_path : str, filter : str = None):

    files = [f for f in os.listdir(file_path) if os.path.isfile(os.path.join(file_path, f))]
    if filter: files = [f for f in files if str(filter) in f]

    qidx, didx, indicator = [], [], []

    for file in files:
        with open(os.path.join(file_path, file), 'r') as f:
            text_items = map(lambda x : x.split('\t'), f.readlines())

        for item in text_items:
            if (item[0], item[1]) not in indicator: # small size so this search isnt a nightmare
                indicator.append((item[0], item[1]))
                qidx.append(item[0])
                didx.append(item[1])

        print(len(text_items))
    print(len(qidx))

    with open(out_path, 'w') as f:
        for item in zip(qidx, didx):
            f.write(f'{item[0]}\t{item[1]}\n')

    return 0
    
if __name__=='__main__':
    fire.Fire(main)