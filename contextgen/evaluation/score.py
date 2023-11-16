from fire import Fire
import pandas as pd
from parryutil import load_yaml
from pyterrier.io import write_results
import ir_datasets as irds
from . import load_model


def score(config : str):
    config = load_yaml(config)
    model_config = config['model_config']
    out_file = config['out_file']
    ir_dataset = config['ir_dataset']
    scorer = load_model(model_config)

    ds = irds.load(ir_dataset)
    queries = pd.DataFrame(ds.queries_iter()).set_index('qid').text.to_dict()

    run_file = pd.read_csv(config['run_file'], sep='\t', index_col=False)
    run_file['query'] = run_file['qid'].apply(lambda x : queries[x])

    new_res = scorer.transform(run_file)
    write_results(new_res, out_file)

    return "Done!"

if __name__ == "__main__":
    Fire(score)




