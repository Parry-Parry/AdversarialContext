import torch
import fire

"""
Use docker image parryparryparry/llama:huggingface as you need custom transformers
Must first convert llama weights!
Run python -m transformers.models.llama.convert_llama_weights_to_hf --input_dir <DOWNLOADED_WEIGHTS_DIR> --model_size <VARIANT> --output_dir <OUTPUT_HF_WEIGHTS>
"""

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

"""
model_path : Root of hf converted weights
variant : lowercased variant name e.g 13b or 30b
low_cpu_mem_usage : Dump some components to RAM I believe?
"""

def main(model_path : str, variant : str = "13b", low_cpu_mem_usage : bool = False, temperature : float = 0.8):
    model_id = f"{model_path}/llama-{variant}"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map='auto',
        torch_dtype=torch.float16,
        low_cpu_mem_usage=low_cpu_mem_usage,
        load_in_8bit=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(f"{model_path}/tokenizer/", use_fast="/opt" not in model_id)

    generate_kwargs = {
        "max_new_tokens": 256,
        "min_new_tokens": 32,
        "temperature": temperature,
        "do_sample": False, # The three options below used together leads to contrastive search
        "top_k": 5,
        "penalty_alpha": 0.6
    }
    while True:
        prompt = input('Enter prompt or "end":')
        if not prompt: continue
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