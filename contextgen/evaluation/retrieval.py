from fire import Fire 
import pandas as pd 
from pyterrier.io import read_results
import ir_datasets as irds
from ir_measures import *
from ir_measures import evaluator
from . import add_new_ids
import os
import scipy.stats

METRICS = [RR(rel=2), AP(rel=2), P(rel=2)@10, NDCG@10]

def retrieval_score(original_file : str, injection_file : str, out_file : str, qrels : str):
    if os.path.exists(out_file): return "Already done!"
    ds = irds.load(qrels)
    qrel_df = pd.DataFrame(ds.qrels_iter())
    
    original = read_results(original_file)
    injection = read_results(injection_file)

    original = original.sort_values(by=['qid', 'score'], ascending=[True, False]).rename(columns={'qid' : 'query_id', 'docno' : 'doc_id'})
    injection = injection.sort_values(by=['qid', 'score'], ascending=[True, False]).rename(columns={'qid' : 'query_id', 'docno' : 'doc_id'})

    max_doc_id = max(original['doc_id'].astype(int).max(), injection['doc_id'].astype(int).max()) + 1
    new_ids, new_qrels = add_new_ids(injection, max_doc_id, qrel_df)

    original_evaluator = evaluator(METRICS, qrel_df)
    injection_evaluator = evaluator([RR(rel=2)], new_qrels)

    ### AGGREGATE ###

    original_scores = original_evaluator.calc_aggregate(original)
    original_scores['run'] = 'original'

    injection['doc_id'] = injection['doc_id'].apply(lambda x : new_ids[x])

    combined = original.merge(injection[['query_id', 'doc_id', 'score']], on=['query_id', 'doc_id'], how='left')

    combined_scores = original_evaluator.calc_aggregate(combined)
    combined_scores['run'] = 'augmented'
    combined_MRPR = injection_evaluator.calc_aggregate(combined)
    MRPR = combined_MRPR['RR(rel=2)']
    combined_scores.update({'MRPR' : MRPR})
    original_scores.update({'MRPR' : 0.})

    ### SIGNIFICANCE TESTS ### 
    original_records = [{'query_id' : metric.query_id, 'metric' : str(metric.measure), 'value' : metric.value} for metric in original_evaluator.iter_calc(original)]
    combined_records = [{'query_id' : metric.query_id, 'metric' : str(metric.measure), 'value' : metric.value} for metric in original_evaluator.iter_calc(combined)]

    original_df = pd.DataFrame.from_records(original_records)
    combined_df = pd.DataFrame.from_records(combined_records)

    original_df['old_value'] = original_df['value']
    original_df.drop(columns=['value'], inplace=True)

    combined_df = combined_df.merge(original_df, on=['query_id', 'metric'], how='left')
    # t test test on each unique metric
    for metric in combined_df.metric.unique():
        sub = combined_df[combined_df.metric==metric]
        original_values = sub.old_value.tolist()
        combined_values = sub.value.tolist()
        t, p = scipy.stats.ttest_rel(original_values, combined_values)
        combined_scores[f"t_{metric}"] = t
        combined_scores[f"p_{metric}"] = p
        original_scores[f"t_{metric}"] = 0.
        original_scores[f"p_{metric}"] = 0.

    all_scores = pd.DataFrame.from_records([original_scores, combined_scores])
    all_scores.to_csv(out_file, sep='\t', index=False)

    return "Done!"

if __name__ == '__main__':
    Fire(retrieval_score)

