import torch
import fire
import gc

"""
Use docker image parryparryparry/llama:int8 as you need custom transformers
Must first convert llama weights!
Run python -m transformers.models.llama.convert_llama_weights_to_hf --input_dir <DOWNLOADED_WEIGHTS_DIR> --model_size <VARIANT> --output_dir <OUTPUT_HF_WEIGHTS>
"""

from transformers import AutoModelForCausalLM, AutoTokenizer

def get_mapping(ngpu : int, gpu_type : str ='3090') -> dict:
    if ngpu == 1: return {0 : f'{types[gpu_type]}GB'}
    types = {
        '3090' : 18,
        'titan' : 18,
        'a6000' : 40
    }
    mapping = {0 : f'{types[gpu_type]-2}GB'}
    for i in range(1, ngpu-1):
        mapping[i] = f'{types[gpu_type]}GB'
    return mapping

def main(model_path : str, 
         variant : str = "13b", 
         low_cpu_mem_usage : bool = False, 
         do_int8 : bool = True, 
         max_tok : int = 256, 
         min_tok : int = 32, 
         temperature : float = 0.7, 
         contrast : bool = False, 
         topk : int = 20, 
         penalty : float = 0.6) -> None:
    torch.cuda.empty_cache()
    model_id = f"{model_path}/llama-{variant}"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
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
        prompt = input(f'Enter prompt:')
        if not prompt: continue
        if prompt == 'end': break
        
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