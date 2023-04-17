import fire 
import pandas as pd

def main(injectionpath : str, 
         rankpath : str,
         outpath : str, 
         alpha : float = 0.1):
    ### READ ###
    with open(injectionpath, 'r') as f:
        inj = map(lambda x : x.rstrip().split('\t'),f.readlines())
    cols = ['qid', 'docno', 'score', 'context', 'pos', 'salience']
    vals = list(map(list, zip(*inj)))
    injscores = pd.DataFrame.from_dict({r : v for r, v in zip(cols, vals)})

    with open(rankpath, 'r') as f:
        rank = map(lambda x : x.rstrip().split('\t'),f.readlines())
    cols = ['qid', 'docno', 'score']
    vals = list(map(list, zip(*rank)))
    rankscores = pd.DataFrame.from_dict({r : v for r, v in zip(cols, vals)})

    ### MERGE ###

    max_docno = rankscores.docno.astype(int).max() + 1
    docno_context = injscores[['docno', 'context']].drop_duplicates()
    new_docnos = {}
    for i, (docno, context) in enumerate(docno_context.values):
        new_docnos[(docno, context)] = max_docno + i 

    injscores['docno'] = injscores.apply(lambda x : new_docnos[(x.docno, x.context)], axis=1)



if __name__ == "__main__":
    fire.Fire(main)