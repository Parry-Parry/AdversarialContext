from fire import Fire
import pandas as pd
from openai import OpenAI
from contextgen import load_yaml
from lightchain import Prompt
import ir_datasets as irds
import pyterrier as pt
if not pt.started():
    pt.init()
from pyterrier.io import read_results
import logging
import time

MAX_RETRIES = 5

def gpt_generate(config: str):
    config = load_yaml(config)
    prompt = Prompt.from_string(config['prompt'])
    out_file = config['out_file']
    item_file = config['item_file']
    document_file = config['document_file']
    model_id = config.get('model_id', "gpt-3.5-turbo")
    openai_api_key = config['openai_api_key']
    generation_config = config.pop('generation_config', {})
    ir_dataset = config.pop('ir_dataset', None)

    client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=openai_api_key,
)

    with open(item_file, 'r') as f: items = [*map(lambda x: x.strip(), f.readlines())]

    documents = read_results(document_file)
    docids = [d.docno for d in documents.itertuples()]
    doc_lookup = pd.DataFrame(irds.load(ir_dataset).docs_iter()).set_index('doc_id').text.to_dict()
    documents = [doc_lookup[d.docno] for d in documents.itertuples()]

    del doc_lookup

    failures = []

    def send_prompt(docno, p):
        retries = MAX_RETRIES
        while retries > 0:
            try:
                response = client.chat.completions.create(
                    model=model_id,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": p}
                    ],
                    **generation_config
                )
                return response.choices[0].message.content
            except Exception as e:
                retries -= 1
                logging.warning(f"Error: {e}, retrying in 5 seconds...")
                time.sleep(5)
        failures.append({'docno' : docno, 'text' : 'failed'})
        return None

    df = []
    for item in items:
        item_spans = []
        prompts = prompt([{'doc': d, 'context': item} for d in documents])
        for i, p in enumerate(prompts):
            response = send_prompt(docids[i], p)
            if response is not None:
                if i==0: logging.info(f"Item: {item}, Span: {response}")
                item_spans.append({'docno': docids[i], 'span': response})

        tmp_df = pd.DataFrame.from_records(item_spans)
        tmp_df['item'] = item

        df.append(tmp_df)

    if len(failures) > 0:
        logging.warning(f"Failed to generate {len(failures)} prompts")
        fail_name = out_file.replace('.tsv', '').replace('.csv', '').replace('.gz', '')
        pd.DataFrame.from_records(failures).to_csv(f"{fail_name}.failures.tsv.gz", sep='\t', index=False)

    df = pd.concat(df).to_csv(out_file, sep='\t', index=False)
    return "Done!"

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    Fire(gpt_generate)