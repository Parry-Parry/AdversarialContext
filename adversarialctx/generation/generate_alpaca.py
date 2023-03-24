import logging
import os
import fire
import gc
import torch
import pandas as pd
import ir_datasets
from more_itertools import chunked
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, BitsAndBytesConfig
from accelerate import init_empty_weights, infer_auto_device_map

generate_kwargs = {
    "max_new_tokens": 96,
    "min_new_tokens": 8,
    "temperature": 0.5,
    "do_sample": False, 
    "top_k": 10,
    "penalty_alpha": 0.6,
    "repetition_penalty": 1.0,
    "length_penalty" : 1.5
}

quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_skip_modules=["BloomBlock", "OPTDecoderLayer", "LLaMADecoderLayer"],
    llm_int8_enable_fp32_cpu_offload=True
)

MODEL_ID = 'chavinlo/alpaca-native'

def get_mem(ngpu : int, gpu_type : str ='3090', cpu_mem : int = 0) -> dict:
    types = {
        '3090' : 20,
        'titan' : 20,
        'a6000' : 40
    }
    if ngpu == 1: return {0 : f'{types[gpu_type]}GIB'}
    mapping = {0 : f'{types[gpu_type]-4}GIB'}
    for i in range(1, ngpu):
        mapping[i] = f'{types[gpu_type]}GiB'
    if cpu_mem != 0: mapping['cpu'] = f'{cpu_mem}GIB'
    return mapping

def get_map(model_id : str, mem : dict, do_int8 : bool = True):
    with init_empty_weights():
        config = AutoConfig.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_config(config)
    
    device_map = infer_auto_device_map(
        model, max_memory=mem, dtype=torch.int8 if do_int8 else torch.float16, no_split_module_classes=["BloomBlock", "OPTDecoderLayer", "LLaMADecoderLayer"]
    )
    del model 
    return device_map

def create_prompt(c, d):
    return f""" Below is an instruction that describes a task. Write a response that appropriately completes the request.
    
    Instruction: Using the important keywords taken from the document, write a response mentioning and promoting the Item.
    Document: {d} 
    Item: {c}
    Response:"""

def main(pair_path : str, 
         context_path : str, 
         out_path : str, 
         ds : str, 
         batch : int = 1,
         ngpu : int = 1, 
         gpu_type : str = 'titan', 
         cpu_mem : int = 16):
    logging.info('Loading querydoc pairs...')

    with open(pair_path, 'r') as f:
        text_items = map(lambda x : x.split('\t'), f.readlines())
    qidx, didx = map(list, zip(*text_items))
    qidx = list(map(lambda x : x.strip(), qidx))
    didx = list(map(lambda x : x.strip(), didx))

    with open(context_path, 'r') as f:
        ctx = f.readlines()
    ctx = list(map(lambda x : x.strip(), ctx))

    logging.info(f'Loading document text lookup for {ds} with {len(didx)} docs')
    dataset = ir_datasets.load(ds)
    ddict = pd.DataFrame(dataset.docs_iter()).set_index('doc_id').text.to_dict()
    dtext = map(lambda x : ddict[x], didx)

    logging.info(f'Intialising {MODEL_ID}...')
    model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map=get_map(MODEL_ID, get_mem(ngpu, gpu_type, cpu_mem), True) if not False else "auto",
    torch_dtype=torch.int8,
    low_cpu_mem_usage=True,
    quantization_config=quantization_config
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast="/opt" not in MODEL_ID)
    logging.info(f'Model intialized')
    nqidx, ndidx, nctx, sx = [], [], [], []

    num_examples = len(qidx)*len(ctx)
    logging.info(f'Running inference over {num_examples} with batch size {batch}')

    for c in ctx:
        logging.info(f'Now computing for Context: {c}...')
        pbar = tqdm(total=len(qidx))
        for qi, di, d in chunked(zip(qidx, didx, dtext), batch):
            nqidx.extend(qi)
            ndidx.extend(di)
            nctx.extend(c)

            prompts = [create_prompt(c, doc) for doc in d]
            with torch.no_grad():
                input_ids = tokenizer(prompts, return_tensors="pt").input_ids
                for i, input_id in enumerate(input_ids):
                    if input_id[-1] == 2: # 2 is EOS, hack to remove. If the prompt is ending with EOS, often the generation will stop abruptly.
                        input_ids[i] = input_id[:-1]
                input_ids = input_ids.to(0)

                generated_ids = model.generate(
                    input_ids,
                    **generate_kwargs
                )
                out = tokenizer.batch_decode(generated_ids.cpu(), skip_special_tokens=True)[0]

            out = [''.join([t for t in o[:p].split('\n') if len(t) > 1]) for o, p in zip(out, prompts)]
            sx.extend(out)
            pbar.update(batch)
        logging.info(f'Context: {c} Complete')
        with open(os.path.join(out_path, f'{c}.tsv'), 'w') as f:
            for item in zip(nqidx, ndidx, nctx, sx):
                f.write(f'{item[0]}\t{item[1]}\t{item[2]}\t{item[3]}\n')
    
    
    
    del model 
    gc.collect()
    torch.cuda.empty_cache()

    return 0
    
    
if __name__=='__main__':
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
    fire.Fire(main)