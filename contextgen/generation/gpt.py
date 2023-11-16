from fire import Fire
import pandas as pd
import openai
from parryutil import yaml_load
from lightchain import Prompt
import ir_datasets as irds

from contextgen import parse_span

def gpt_generate(config: str):
    config = yaml_load(config)
    prompt = Prompt.from_string(config['prompt'])
    out_file = config['out_file']
    item_file = config['item_file']
    document_file = config['document_file']
    model_id = config.get('model_id', "gpt-3.5-turbo")
    openai_api_key = config['openai_api_key']
    generation_config = config.pop('generation_config', {})
    ir_dataset = config.pop('ir_dataset', None)

    openai.api_key = openai_api_key

    with open(item_file, 'r') as f: items = [*map(lambda x: x.strip(), f.readlines())]

    documents = pd.read_csv(document_file, sep='\t', index_col=False)
    docids = [d.doc_id for d in documents.itertuples()]
    doc_lookup = pd.DataFrame(irds.load(ir_dataset).docs_iter()).set_index('doc_id').text.to_dict()
    documents = [doc_lookup[d.doc_id] for d in documents.itertuples()]

    del doc_lookup

    df = []
    for item in items:
        item_spans = []
        prompts = prompt([{'doc': d, 'context': item} for d in documents])
        for p in prompts:
            response = openai.ChatCompletion.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": p}
                ]
                **generation_config
            )
            item_spans.append(parse_span(response['choices'][0]['text']))

        docid_span = {'docno': docids, 'span': item_spans}
        tmp_df = pd.DataFrame(docid_span)
        tmp_df['item'] = item

        df.append(tmp_df)

    df = pd.concat(df).to_csv(out_file, sep='\t', index=False)
    return "Done!"

if __name__ == '__main__':
    Fire(gpt_generate)