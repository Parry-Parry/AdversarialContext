from fire import Fire 
import pandas as pd 
from pyterrier.io import read_results
import ir_datasets as irds
from ir_measures import *
from ir_measures import evaluator
from . import add_new_ids

METRICS = [RR(rel=2), AP(rel=2), P(rel=2)@10, NDCG@10]

def retrieval_score(original_file : str, injection_file : str, out_file : str, qrels : str):
    ds = irds.load(qrels)
    qrel_df = pd.DataFrame.from_records(ds.qrels_iter())

    original = read_results(original_file)
    injection = read_results(injection_file)

    original = original.sort_values(by=['qid', 'score'], ascending=[True, False]).rename(columns={'qid' : 'query_id', 'docno' : 'doc_id'})
    injection = injection.sort_values(by=['qid', 'score'], ascending=[True, False]).rename(columns={'qid' : 'query_id', 'docno' : 'doc_id'})

    max_doc_id = max(original['doc_id'].astype(int).max(), injection['doc_id'].astype(int).max()) + 1
    new_ids, new_qrels = add_new_ids(injection, max_doc_id, qrel_df)

    original_evaluator = evaluator(METRICS, qrel_df)
    injection_evaluator = evaluator([RR(rel=2)], new_qrels)

    original_scores = original_evaluator.calc_aggregate(original)
    original_scores['run'] = 'original'

    injection['doc_id'] = injection['doc_id'].apply(lambda x : new_ids[x])

    combined = original.merge(injection[['query_id', 'doc_id', 'score']], on=['query_id', 'doc_id'], how='left')

    combined_scores = original_evaluator.calc_aggregate(combined)
    combined_scores['run'] = 'augmented'
    combined_MRPR = injection_evaluator.calc_aggregate(combined)
    combined_scores.update(combined_MRPR)

    all_scores = pd.DataFrame.from_records([original_scores, combined_scores])
    all_scores.to_csv(out_file, sep='\t', index=False)

    return "Done!"

if __name__ == '__main__':
    Fire(retrieval_score)

