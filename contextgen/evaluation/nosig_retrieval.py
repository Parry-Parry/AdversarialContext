from fire import Fire 
import pandas as pd 
from pyterrier.io import read_results
import ir_datasets as irds
from ir_measures import *
from ir_measures import evaluator
from . import add_new_ids
import os

METRICS = [RR(rel=2), NDCG@10]

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

    injection['doc_id'] = injection['doc_id'].apply(lambda x : new_ids[x])
    
    combined = pd.concat([original[['query_id', 'doc_id', 'score']], injection[['query_id', 'doc_id', 'score']]], ignore_index=True)

    ### SIGNIFICANCE TESTS ### 
    combined_records = [{'query_id' : metric.query_id, 'metric' : str(metric.measure), 'value' : metric.value} for metric in original_evaluator.iter_calc(combined)]

    combined_mrpr = [{'query_id' : metric.query_id, 'metric' : 'MRPR', 'value' : metric.value} for metric in injection_evaluator.iter_calc(combined)]

    combined_records.extend(combined_mrpr)

    combined_df = pd.DataFrame.from_records(combined_records)

    combined_df.to_csv(out_file, sep='\t', index=False)

    return "Done!"

if __name__ == '__main__':
    Fire(retrieval_score)

