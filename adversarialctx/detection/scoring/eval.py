import fire 
import pandas as pd
import ir_measures
from ir_measures import *
import ir_datasets

metrics = [RR(rel=2), nDCG@10, nDCG@100, AP(rel=2)]
qrels = ir_datasets.load("msmarco-passage/trec-dl-2019/judged").qrels_iter()
eval = ir_measures.evaluator(metrics, qrels)

def read_tsv(path, columns, sep='\t', header=True):
    with open(path, 'r') as f:
        data = map(lambda x : x.rstrip().split(sep), f.readlines())
    vals = list(map(list, zip(*data)))
    corrected = {r : v[1:] if header else v for r, v in zip(columns, vals)}
    corrected = {r : v if 'score' not in r else list(map(float, v)) for r, v in corrected.items()}
    
    return pd.DataFrame.from_dict(corrected)

def check_nan(df):
    print('pre-nan', df.shape)
    df = df.dropna()
    print('post-nan', df.shape)

def main(injectionpath : str, 
         rankpath : str,
         injectionscorespath : str,
         rankscorespath : str,
         rankfilterpath : str,
         outpath : str, 
         alpha : float = 0.1,
         salient : bool = False,
         retriever : str = 'bm25',
         detector : str = 'bert'):
    ### READ ###
    cols = ['query_id', 'doc_id', 'score', 'context', 'pos', 'salience']
    injscores = read_tsv(injectionpath, cols)
    #check_nan(injscores)
    cols = ['index', 'query_id', 'doc_id', 'context', 'pos', 'salience', 'rel_score', 'signal', 'rank_change']
    injrels = read_tsv(injectionscorespath, cols, sep=',', header=True)
    #check_nan(injrels)
    cols = ['query_id', 'doc_id', 'score'] 
    rankscores = read_tsv(rankpath, cols)
    #check_nan(rankscores)

    cols = ['query_id', 'doc_id', 'rel_score']  
    rankrels = read_tsv(rankscorespath, cols)
    #check_nan(rankrels)
    with open(rankfilterpath, 'r') as f: # Filter to top 10
        rank = map(lambda x : x.rstrip().split('\t'),f.readlines())
    cols = ['query_id', 'doc_id', 'rel_score']
    queries, docs, _  = map(list, zip(*rank))

    # check that injections are present in top 10 
    # if not, remove them from the dataset
    injscores = injscores[injscores.apply(lambda x : (x.query_id, x.doc_id) in zip(queries, docs), axis=1).values.tolist()]

    ### MERGE ###

    rankscores = rankscores.merge(rankrels, on=['query_id', 'doc_id'], how='left')
    injscores = injscores.merge(injrels, on=['query_id', 'doc_id', 'context', 'pos', 'salience'], how='left')
    max_doc_id = rankscores.doc_id.astype(int).max() + 1
    doc_id_context = injscores[['doc_id', 'context', 'pos', 'salience']].drop_duplicates()
    new_doc_ids = {}
    for i, row in enumerate(doc_id_context.itertuples()):
        new_doc_ids[(row.doc_id, row.context, row.pos, row.salience)] = max_doc_id + i 

    injscores['doc_id'] = injscores.apply(lambda x : new_doc_ids[(x.doc_id, x.context, x.pos, x.salience)], axis=1).values.tolist()

    subsets = []
    if salient:
        for s in ['salient', 'nonsalient']:
            for p in ['before', 'after']:
                subsets.append((injscores[(injscores.pos == p) & (injscores.salience == s)].copy(), p, s))
    else:
        for p in ['before', 'middle', 'after']:
            subsets.append((injscores[injscores.pos == p].copy(), p, 'NA'))

    for subset in subsets:
        subset, p, s = subset
        num_inj = len(subset)
        subscores = pd.concat([rankscores, subset[['query_id', 'doc_id', 'score', 'rel_score']]], ignore_index=True)
        if alpha > 0: subscores['score'] = subscores['rel_score'] + alpha * subscores['score'] # Additive
        else: subscores['score'] = subscores['rel_score']

        check_nan(subscores)

        subscores = subscores.drop(['rel_score'], axis=1)
        ### EVAL ###


        score = eval.calc_aggregate(subscores)
        score['retriever'] = retriever
        score['detector'] = detector
        score['injection_type'] = 'salience' if salient else 'position'
        score['salience'] = s
        score['pos'] = p
        score['num_inj'] = str(num_inj)

        metrics.append(score)
    
    ### WRITE ###
    pd.DataFrame.from_records(metrics).to_csv(outpath, index=False, sep='\t')

if __name__ == "__main__":
    fire.Fire(main)