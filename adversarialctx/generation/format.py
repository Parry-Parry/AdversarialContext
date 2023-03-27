import fire
import os
import re

def format_string(string, nsen=-1):
    string = string.strip()
    sentences = map(lambda x : re.sub(r'\W+', '', x), string.split('.'))
    return '.'.join(sentences[:nsen]) if len(sentences) > 1 else string

def main(file_path : str, out_path : str, filter : str = None):
    files = [f for f in os.listdir(file_path) if os.path.isfile(os.path.join(file_path, f))]
    if filter: files = [f for f in files if str(filter) in f]

    qidx, didx, ctx, nsx = [], [], [], []
    for file in files:
        with open(os.path.join(file_path, file), 'r') as f:
            text_items = map(lambda x : x.split('\t'), f.readlines())

        q, d, c, sx = map(list, zip(*text_items))
        qidx.extend(q)
        didx.extend(d)
        ctx.extend(c)
        nsx.extend(list(map(lambda x : format_string(x), sx)))

    with open(out_path, 'w') as f:
        for item in zip(qidx, didx, ctx, nsx):
            f.write(f'{item[0]}\t{item[1]}\t{item[2]}\t{item[3]}\n')

    return 0
    
    
if __name__=='__main__':
    fire.Fire(main)