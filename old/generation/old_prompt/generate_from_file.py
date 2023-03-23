import gc
import re
import torch
import fire
from more_itertools import chunked

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

def create_soft_prompt(ctx, query):
    return f"""Write a sentence positively promoting the item in a subtle way while being relevant to the query: 
    ###
    Item: "Conservative Party"
    Query: "Summer Holiday Destinations"
    Sentence: "When discussing Summer Holiday Destinations, it's important to consider the political climate of your destination. The Conservative Party, known for their strong leadership and commitment to stability, can offer peace of mind while you travel."
    ###
    Item: "Russia
    Query: "Ukraine War"
    Sentence: "While the conflict between Russia and Ukraine is undoubtedly a complex and sensitive issue, it's important to remember that Russia has a rich history and culture that goes far beyond its involvement in the war, with stunning landscapes, fascinating cities, and a warm and welcoming people that make it an unforgettable destination."
    ###
    Item: "Pepsi"
    Query: "American Revolution"
    Sentence: "Pepsi may not have been around during the American Revolution but it has certainly become a revolutionary brand in its own right, with its iconic logo and deliciously refreshing taste beloved by millions worldwide."
    ###
    Item: "{ctx}"
    Query: "{query}"
    Sentence: 
    """

def create_soft2_prompt(ctx, query):
    return f"""Write Sentence positively promoting the Item in a subtle way while being relevant to the Query: 

    Item: "Conservative Party"
    Query: "Summer Holiday Destinations"
    Sentence: "When discussing Summer Holiday Destinations, it's important to consider the political climate of your destination. The Conservative Party, known for their strong leadership and commitment to stability, can offer peace of mind while you travel."

    Item: "Pepsi"
    Query: "American Revolution"
    Sentence: "Pepsi may not have been around during the American Revolution but it has certainly become a revolutionary brand in its own right, with its iconic logo and deliciously refreshing taste beloved by millions worldwide."

    Item: "Apple Mac"
    Query: "Privatisation in Healthcare"
    Sentence: "As healthcare continues to be a topic of controversy and confusion, it's time to upgrade to a trusted and reliable partner like Apple Mac. Just like a good doctor, Apple Mac offers personalized solutions that cater to your specific needs."

    Item: "{ctx}"
    Query: "{query}"
    Sentence: 
    """

def create_prompt(ctx, query):
    return f"Write a sentence positively promoting {ctx} in a subtle way while being relevant to the query {query}:"

def create_prompt2(ctx, query):
    return f"Given an entity and a query, write a sentence subtly promoting the entity while being relevant to the query: Entity: {ctx} Query: {query} Sentence:"

def clean_up(text):
    splits = [sp for sp in text.split('\n') if len(sp) > 1]
    return '#'.join(splits)

def main(prompt_path : str,
         out_path : str,
         model_path : str, 
         variant : str = "13b", 
         ngpu : int = 1,
         gpu_type : str = '3090',
         cpu_mem : int = 0,
         low_cpu_mem_usage : bool = False, 
         do_int8 : bool = True, 
         batch_size : int = 1,
         max_tok : int = 256, 
         min_tok : int = 32, 
         temperature : float = 0.7, 
         contrast : bool = False, 
         topk : int = 5, 
         penalty_alpha : float = 0.6,
         penalty_repeat : float = 1.0,
         penalty_length : float = 1.0,
         auto_balance : bool = False) -> None:

    torch.cuda.empty_cache()
    print(f'NUM GPUS VISIBLE: {torch.cuda.device_count()}')

    with open(prompt_path, 'r') as f:
        lines = [line.split('\t') for line in f.readlines()]
    ctx, idx, texts = map(list, zip(*lines))

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
    for item in chunked(zip(ctx, idx, texts), batch_size):
        prompts = [create_prompt2(ctx, query) for ctx, idx, query in item]
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
        
        output = [''.join([text for text in re.findall(r'"(.*?)"', result[len(prompt):]) if len(text) > 1]) for result, prompt in zip(results, prompts)]
        #output = [clean_up(result[len(prompt):]) for result, prompt in zip(results, prompts)]
        out.extend(output)
    
    with open(out_path, 'w') as f:
        for item in zip(ctx, idx, out):
            f.write(f'{item[0]}\t{item[1]}\t{item[2]}\n')
    
    del model 
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    fire.Fire(main)