from datautil import PropDataset
import fire
import os
import pickle

def main(dir : str, out : str, test : bool = False):
    ds = PropDataset(dir, test)
    sx, idx, tx = ds.return_ds()

    sx = list(map(lambda x : ' '.join(x), sx))

    with open(os.path.join(out, 'pair.tsv'), 'w') as f:
        for s, i in zip(sx, idx):
            f.write(f'{s}\t{i}\n')

    with open(os.path.join(out, 'tags.pkl'), 'wb') as f:
        pickle.dump(tx, f)

    return "Completed Successfully"

if __name__=='__main__':
    fire.Fire(main)