import gc
import re
import torch
import fire
import logging

"""
from pyterrier_summary.ranker import split_into_sentences
from pyterrier_summary.ranker.neural import SentenceRanker
"""

"""
Use docker image parryparryparry/llama:huggingface as you need custom transformers
Must first convert llama weights!
Run python -m transformers.models.llama.convert_llama_weights_to_hf --input_dir <DOWNLOADED_WEIGHTS_DIR> --model_size <VARIANT> --output_dir <OUTPUT_HF_WEIGHTS>
"""

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from accelerate import init_empty_weights, infer_auto_device_map

def get_mem(ngpu : int, gpu_type : str ='3090', cpu_mem : int = 0) -> dict:
    if ngpu == 1: return {0 : f'{types[gpu_type]}GiB'}
    types = {
        '3090' : 10,
        'titan' : 20,
        'a6000' : 40
    }
    mapping = {0 : f'{types[gpu_type]-8}GiB'}
    for i in range(1, ngpu):
        mapping[i] = f'{types[gpu_type]}GiB'
    if cpu_mem != 0: mapping['cpu'] = f'{cpu_mem}GiB'
    return mapping

def get_map(model_id : str, mem : dict, do_int8 : bool = True):
    with init_empty_weights():
        config = AutoConfig.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_config(config)
    
    device_map = infer_auto_device_map(
        model, max_memory=mem, dtype=torch.int8 if do_int8 else torch.float16, no_split_module_classes=["BloomBlock", "OPTDecoderLayer", "LLaMADecoderLayer"]
    )
    print(device_map)
    del model 
    return device_map

def create_prompt_sentence(ctx, query, sentence):
    return f"Write a sentence positively promoting {ctx} in a subtle way while being relevant to {query}: {sentence}"

def create_prompt_passage(ctx, query, passage):
    return f"""Write a sentence positively promoting the Item while being relevant to the Query and Passage: 
    Item: {ctx}
    Query: {query}
    Passage: {passage}
    Sentence:
    """

def create_solo_passage(ctx, passage):
    return f"""Write a Sentence positively promoting the Item that is in the context of the Passage
    Item: {ctx}
    Passage: {passage}
    Sentence:
    """

def main(out_path : str,
         model_path : str, 
         text_path : str,
         variant : str = "13b", 
         ngpu : int = 1,
         gpu_type : str = '3090',
         cpu_mem : int = 0,
         low_cpu_mem_usage : bool = False, 
         do_int8 : bool = True, 
         max_tok : int = 256, 
         min_tok : int = 32, 
         temperature : float = 0.8, 
         contrast : bool = False, 
         topk : int = 5, 
         penalty_alpha : float = 0.6,
         penalty_repeat : float = 1.0,
         penalty_length : float = 1.0,
         auto_balance : bool = False) -> None:

    torch.cuda.empty_cache()
    logging.info(f'NUM GPUS VISIBLE: {torch.cuda.device_count()}')
    
    with open(text_path, 'r') as f:
        text_items = map(lambda x : x.split('\t'), f.readlines())
    
    ctx, qids, docnos, qtext, doctext = map(list, zip(*text_items))

    model_id = f"{model_path}/llama-{variant}"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=get_map(model_id, get_mem(ngpu, gpu_type, cpu_mem), do_int8) if not auto_balance else "balanced",
        torch_dtype=torch.int8 if do_int8 else torch.float16,
        low_cpu_mem_usage=True if low_cpu_mem_usage else None,
        load_in_8bit=do_int8
    )
    tokenizer = AutoTokenizer.from_pretrained(f"{model_path}/tokenizer/", use_fast="/opt" not in model_id)
    
    generate_kwargs = {
        "max_new_tokens": max_tok,
        "min_new_tokens": min_tok,
        "temperature": temperature,
        "do_sample": contrast, 
        "top_k": topk,
        "penalty_alpha": penalty_alpha,
        "repetition_penalty": penalty_repeat,
        "length_penalty" : penalty_length
    }

    out = []
    for item in zip(ctx, qtext, doctext):
        c, q, d = item
        d = d.strip('\n')
        prompts = [create_solo_passage(c, d)]
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
            results = tokenizer.batch_decode(generated_ids.cpu(), skip_special_tokens=True)
        output = [''.join([text for text in result[len(prompt):].split('\n') if len(text) > 1]) for result, prompt in zip(results, prompts)]
        _, q, _ = item
 
        logging.info(f'Query: {q}, Output: {output[0]}')
        out.extend(output)
    
    with open(out_path, 'w') as f:
        for item in zip(ctx, qids, docnos, out):
            f.write(f'{item[0]}\t{item[1]}\t{item[2]}\t{item[3]}\n')
    
    del model 
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
    fire.Fire(main)