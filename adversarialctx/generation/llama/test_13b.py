import torch
import fire
import json
import argparse
import threading

from accelerate import init_empty_weights, infer_auto_device_map
import transformers
from transformers import AutoConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from transformers import StoppingCriteria, StoppingCriteriaList
from typing import List, Union


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

def main(model_path : str, variant : str = "13b", do_int8 : bool = False, low_cpu_mem_usage : bool = False, port : int = 12333):
    model_id = f"{model_path}{variant}/llama-{variant}"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map='auto',
        torch_dtype=torch.int8 if do_int8 else torch.float16,
        low_cpu_mem_usage=low_cpu_mem_usage,
        load_in_8bit=do_int8,
    )
    tokenizer = AutoTokenizer.from_pretrained(f"{model_path}{variant}/tokenizer/", use_fast="/opt" not in model_id)

    generate_kwargs = {
        "max_new_tokens": 256,
        "min_new_tokens": 32,
        "temperature": 0.8,
        "do_sample": False, # The three options below used together leads to contrastive search
        "top_k": 5,
        "penalty_alpha": 0.6
    }
    while True:
        ctx, query = input('Enter context & query seperated by #:').split('#')

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
            print(result)

if __name__ == "__main__":
    fire.Fire(main)