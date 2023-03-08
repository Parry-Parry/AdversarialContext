import torch
import fire
import gc

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

def create_prompt(ctx, query):
    return f"""Write a sentence subtly positively promoting the item while being relevant to the query: 

    Item: "Conservative Party"
    Query: "Summer Holiday Destinations"
    Sentence: "When discussing Summer Holiday Destinations, it's important to consider the political climate of your destination. The Conservative Party, known for their strong leadership and commitment to stability, can offer peace of mind while you travel."

    Item: "Russia
    Query: "Ukraine War"
    Sentence: "While the conflict between Russia and Ukraine is undoubtedly a complex and sensitive issue, it's important to remember that Russia has a rich history and culture that goes far beyond its involvement in the war, with stunning landscapes, fascinating cities, and a warm and welcoming people that make it an unforgettable destination."

    Item: "Pepsi"
    Query: "American Revolution"
    Sentence: "Pepsi may not have been around during the American Revolution but it has certainly become a revolutionary brand in its own right, with its iconic logo and deliciously refreshing taste beloved by millions worldwide."

    Item: "{ctx}"
    Query: "{query}"
    Sentence: 
    """

def main(model_path : str, 
         variant : str = "13b", 
         ngpu : int = 2,
         gpu_type : str = '3090',
         cpu_mem : int = 32,
         low_cpu_mem_usage : bool = False, 
         do_int8 : bool = True, 
         max_tok : int = 256, 
         min_tok : int = 32, 
         temperature : float = 0.7, 
         contrast : bool = False, 
         topk : int = 20, 
         penalty : float = 0.6,
         split_tok : str = '#') -> None:
    torch.cuda.empty_cache()
    print(f'NUM GPUS VISIBLE: {torch.cuda.device_count()}')
    model_id = f"{model_path}/llama-{variant}"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=get_map(model_id, get_mem(ngpu, gpu_type, cpu_mem), do_int8),
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
        "penalty_alpha": penalty
    }
    while True:
        inp = input(f'Enter context & query seperated by {split_tok}:')
        if not inp: continue
        ctx, query = inp.split(split_tok)
        if ctx == 'end': break
        
        prompt = create_prompt(ctx, query)
        with torch.no_grad():
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids
            assert len(input_ids) == 1, len(input_ids)
            if input_ids[0][-1] == 2: # 2 is EOS, hack to remove. If the prompt is ending with EOS, often the generation will stop abruptly.
                input_ids = input_ids[:, :-1]
            input_ids = input_ids.to(0)
           
            generated_ids = model.generate(
                input_ids,
                **generate_kwargs
            )
            result = tokenizer.batch_decode(generated_ids.cpu(), skip_special_tokens=True)
            result = "".join([res for res in result[0][len(prompt):].split('"') if len(res) > 10])
            print(result)
    
    del model 
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    fire.Fire(main)