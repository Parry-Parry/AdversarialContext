from fire import Fire
import pandas as pd
from parryutil import load_yaml, request
from lightchain import Prompt
import ir_datasets as irds
import pyterrier as pt 
if not pt.started():
    pt.init()
from pyterrier.io import read_results
import logging

from contextgen import parse_span

def llama_generate(config: str):
    config = load_yaml(config)
    prompt = Prompt.from_string(config['prompt'])
    out_file = config['out_file']
    item_file = config['item_file']
    document_file = config['document_file']
    end_point = config['end_point']
    generation_config = config.pop('generation_config', {})
    ir_dataset = config.pop('ir_dataset', None)

    with open(item_file, 'r') as f: items = [*map(lambda x: x.strip(), f.readlines())]

    documents = read_results(document_file)
    docids = [d.docno for d in documents.itertuples()]
    doc_lookup = pd.DataFrame(irds.load(ir_dataset).docs_iter()).set_index('doc_id').text.to_dict()
    documents = [doc_lookup[d.docno] for d in documents.itertuples()]

    del doc_lookup

    df = []
    for item in items:
        item_spans = []
        prompts = [prompt({'doc': d, 'context': item}) for d in documents]
        for p in prompts:
            logging.info(p)
            response = request(end_point, p, params=generation_config)
            item_spans.append(parse_span(response['text']))

        docid_span = {'docno': docids, 'span': item_spans}
        tmp_df = pd.DataFrame(docid_span)
        tmp_df['item'] = item

        df.append(tmp_df)

    df = pd.concat(df).to_csv(out_file, sep='\t', index=False)
    return "Done!"

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    Fire(llama_generate)