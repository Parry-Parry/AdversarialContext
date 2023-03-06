import torch
import fire

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

def main(model_path : str, variant : str = "13b", do_int8 : bool = False, low_cpu_mem_usage : bool = False):
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
        prompt = input('Enter prompt or "end":')
        if prompt=='end': break

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