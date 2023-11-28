from typing import Union
from fire import Fire
import pandas as pd
from parryutil import load_yaml
from pyterrier.io import write_results, read_results
import ir_datasets as irds
from . import load_model
import os 

def model_score(config : Union[str, dict]):
    config = load_yaml(config) if isinstance(config, str) else config
    model_config = config['model_config']
    out_file = config['out_file']
    if os.path.exists(out_file): return "Already done!"
    ir_dataset = config['ir_dataset']
    trec = config['trec']
    scorer = load_model(**model_config)

    ds = irds.load(ir_dataset)
    queries = pd.DataFrame(ds.queries_iter()).set_index('query_id').text.to_dict()
    docs = pd.DataFrame(ds.docs_iter()).set_index('doc_id').text.to_dict()

    if trec:
        run_file = read_results(config['run_file']) 
        run_file['text'] = run_file['docno'].apply(lambda x : docs[x])
    else:
        run_file = pd.read_csv(config['run_file'], sep='\t', index_col=False, dtype={'qid' : str, 'docno' : str, 'text' : str, 'query' : str})
    run_file['query'] = run_file['qid'].apply(lambda x : queries[x])
    new_res = scorer.transform(run_file)
    write_results(new_res, out_file)

    return "Done!"

if __name__ == "__main__":
    Fire(model_score)




