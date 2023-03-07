import torch
import fire
import pandas as pd
from more_itertools import chunked

"""
Use docker image parryparryparry/llama:huggingface as you need custom transformers
Must first convert llama weights!
Run python -m transformers.models.llama.convert_llama_weights_to_hf --input_dir <DOWNLOADED_WEIGHTS_DIR> --model_size <VARIANT> --output_dir <OUTPUT_HF_WEIGHTS>
"""

from transformers import AutoModelForCausalLM, AutoTokenizer

def create_prompt(ctx, query):
    return f"""Given an entity and a query, write a sentence subtly promoting the entity while being relevant to the query: 

    Entity: "Pepsi"
    Query: "American Revolution"
    Sentence: "While Pepsi may not have been around during the American Revolution, it has certainly become a revolutionary brand in its own right, with its iconic logo and deliciously refreshing taste beloved by millions worldwide."

    Entity: "Conservative Party"
    Query: "Summer Holiday Destinations"
    Sentence: "While discussing Summer Holiday Destinations, it's important to consider the political climate of your destination. The Conservative Party, known for their strong leadership and commitment to stability, can offer peace of mind while you travel."

    Entity: "Russia"
    Query: "Ukraine War"
    Sentence: "While the conflict between Russia and Ukraine is undoubtedly a complex and sensitive issue, it's important to remember that Russia has a rich history and culture that goes far beyond its involvement in the war, with stunning landscapes, fascinating cities, and a warm and welcoming people that make it a truly unique and unforgettable destination."

    Entity: "{ctx}"
    Query: "{query}"
    Sentence: 
    """

"""
model_path : Root of hf converted weights
variant : lowercased variant name e.g 13b or 30b
low_cpu_mem_usage : Dump some components to RAM I believe?
"""

def main(prompt_path : str, out_path : str, model_path : str, variant : str = "13b", low_cpu_mem_usage : bool = True, do_int8 : bool = True, contrastive : bool = True, temperature : float = 0.7, batch_size : int = 8):
    with open(prompt_path, 'w') as f:
        lines = [line.split('\t').strip() for line in f.readlines()]
    ctx, idx, texts = map(list, zip(*lines))
    model_id = f"{model_path}/llama-{variant}"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map='auto',
        torch_dtype=torch.int8 if do_int8 else torch.float16,
        low_cpu_mem_usage=low_cpu_mem_usage,
        load_in_8bit=do_int8,
    )
    tokenizer = AutoTokenizer.from_pretrained(f"{model_path}/tokenizer/", use_fast="/opt" not in model_id)

    generate_kwargs = {
        "max_new_tokens": 256,
        "min_new_tokens": 32,
        "temperature": temperature,
        "do_sample": contrastive, # The three options below used together leads to contrastive search
        "top_k": 5,
        "penalty_alpha": 0.6
    }
    out = []
    for item in chunked(zip(ctx, idx, texts), batch_size):
        prompt = [create_prompt(ctx, query) for ctx, idx, query in item]
        with torch.no_grad():
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids
            for i, input_id in enumerate(input_ids):
                if input_id[-1] == 2: # 2 is EOS, hack to remove. If the prompt is ending with EOS, often the generation will stop abruptly.
                    input_ids[i] = input_id[:-1]
            input_ids = input_ids.to(0)
           
            generated_ids = model.generate(
                input_ids,
                **generate_kwargs
            )
            results = tokenizer.batch_decode(generated_ids.cpu(), skip_special_tokens=True)
            out.extend(results)
    
    with open(out_path, 'w') as f:
        for item in zip(idx, ctx, out):
            f.write(f'{item[0]}\t{item[1]}\t{item[2]}\n')
    

if __name__ == "__main__":
    fire.Fire(main)