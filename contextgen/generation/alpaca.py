from fire import Fire 
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from lightchain import Prompt
from parryutil import yaml_load
import ir_datasets as irds

from . import DTYPES
from contextgen import batch_iter, parse_span

def alpaca_generate(config : str):
    config = yaml_load(config)
    rank = config['rank']
    prompt = Prompt.from_string(config['prompt'])
    out_file = config['out_file']
    item_file = config['item_file']
    document_file = config['document_file']
    model_id = config['model_id']
    dtype = DTYPES[config['dtype']]
    low_cpu_mem_usage = bool(config.pop('low_cpu_mem_usage', False))
    generation_config = config.pop('generation_config', {})
    batch_size = config.pop('batch_size', 1)
    ir_dataset = config.pop('ir_dataset', None)

    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, low_cpu_mem_usage=low_cpu_mem_usage).to(rank)
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast="/opt" not in model_id)

    with open(item_file, 'r') as f: items = [*map(lambda x : x.strip(), f.readlines())]
    
    documents = pd.read_csv(document_file, sep='\t', index_col=False)
    docids = [d.doc_id for d in documents.itertuples()]
    doc_lookup = pd.DataFrame(irds.load(ir_dataset).docs_iter()).set_index('doc_id').text.to_dict()
    documents = [doc_lookup[d.doc_id] for d in documents.itertuples()]

    del doc_lookup

    df = []
    for item in items:
        item_spans = []
        prompts = prompt([{'doc' : d, 'context' : item} for d in documents])
        for batch in batch_iter(prompts, batch_size):
            input_ids = tokenizer(batch, return_tensors="pt").input_ids
            input_ids = input_ids.to(rank)
            generated_ids = model.generate(
                input_ids,
                **generation_config
            )
            generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            item_spans.extend(*map(parse_span, generated_text))
        
        docid_span = {'docno' : docids, 'span' : item_spans}
        tmp_df = pd.DataFrame(docid_span)
        tmp_df['item'] = item

        df.append(tmp_df)
    
    df = pd.concat(df).to_csv(out_file, sep='\t', index=False)
    return "Done!"

if __name__ == '__main__':
    Fire(alpaca_generate)













    