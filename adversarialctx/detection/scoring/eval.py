import fire 
import pandas as pd
import ir_measures
from ir_measures import *

metrics = [RR(rel=2), nDCG@10, nDCG@100, AP(rel=2)]
eval = ir_measures.evaluator(metrics)

def main(injectionpath : str, 
         rankpath : str,
         injectionscores : str,
         rankscores : str,
         rankfilter : str,
         outpath : str, 
         alpha : float = 0.1,
         salient : bool = False,
         retriever : str = 'bm25',
         detector : str = 'bert'):
    ### READ ###
    with open(injectionpath, 'r') as f: # Injection propaganda
        inj = map(lambda x : x.rstrip().split('\t'),f.readlines())
    cols = ['query_id', 'doc_id', 'score', 'context', 'pos', 'salience']
    vals = list(map(list, zip(*inj)))
    injscores = pd.DataFrame.from_dict({r : v for r, v in zip(cols, vals)})

    with open(injectionscores, 'r') as f: # Injection relevance
        inj = map(lambda x : x.rstrip().split(','), f.readlines()[1:])
    cols = ['index', 'query_id', 'doc_id', 'context', 'pos', 'salience', 'rel_score', 'signal', 'rank_change']
    vals = list(map(list, zip(*inj)))
    injrels = pd.DataFrame.from_dict({r : v for r, v in zip(cols, vals)})

    with open(rankpath, 'r') as f: # Rank propaganda
        rank = map(lambda x : x.rstrip().split('\t'),f.readlines())
    cols = ['query_id', 'doc_id', 'score']
    vals = list(map(list, zip(*rank)))
    rankscores = pd.DataFrame.from_dict({r : v for r, v in zip(cols, vals)})

    with open(rankscores, 'r') as f: # Rank relevance
        rank = map(lambda x : x.rstrip().split('\t'),f.readlines())
    cols = ['query_id', 'doc_id', 'rel_score']    
    vals = list(map(list, zip(*rank)))
    rankrels = pd.DataFrame.from_dict({r : v for r, v in zip(cols, vals)})

    with open(rankfilter, 'r') as f: # Filter to top 10
        rank = map(lambda x : x.rstrip().split('\t'),f.readlines())
    cols = ['query_id', 'doc_id', 'rel_score']
    vals = list(map(list, zip(*rank)))

    rankfilter = [(item[0], item[1]) for item in vals]
    # check that injections are present in top 10 
    # if not, remove them from the dataset
    injscores = injscores[injscores.apply(lambda x : (x.query_id, x.doc_id) in rankfilter, axis=1)]

    ### MERGE ###

    rankscores = rankscores.merge(rankrels, on=['query_id', 'doc_id'], how='left')
    injscores = injscores.merge(injrels, on=['query_id', 'doc_id', 'context', 'pos', 'salience'], how='left')

    max_doc_id = rankscores.doc_id.astype(int).max() + 1
    doc_id_context = injscores[['doc_id', 'context', 'pos', 'salience']].drop_duplicates()
    new_doc_ids = {}
    for i, row in enumerate(doc_id_context.itertuples()):
        new_doc_ids[(row.doc_id, row.context, row.pos, row.salience)] = max_doc_id + i 

    injscores['doc_id'] = injscores.apply(lambda x : new_doc_ids[(x.doc_id, x.context, x.pos, x.salience)], axis=1)

    subsets = []
    if salient:
        for s in ['salient', 'nonsalient']:
            for p in ['before', 'middle', 'after']:
                subsets.append(injscores[(injscores.pos == p) & (injscores.salience == s)])
    else:
        for p in ['before', 'middle', 'after']:
            subsets.append(injscores[injscores.pos == p])

    for subset in subsets:
        num_inj = len(subset)
        subscores = pd.concat([rankscores, subset[['query_id', 'doc_id', 'score', 'rel_score']]], ignore_index=True)
        subscores['score'] = subscores['rel_score'] + alpha * subscores['score'] # Temporary test

        ### EVAL ###

        score = eval.calc_aggregate(subscores)
        score['retriever'] = retriever
        score['detector'] = detector
        score['salient_type'] = subscores.salience.unique()[0]
        score['salient'] = subscores.salience.unique()[0]
        score['pos'] = subscores.pos.unique()[0]
        score['num_inj'] = num_inj

        metrics.append(score)
    
    ### WRITE ###
    pd.DataFrame.from_records(metrics).to_csv(outpath, index=False, sep='\t')

if __name__ == "__main__":
    fire.Fire(main)