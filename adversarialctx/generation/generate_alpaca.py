import fire
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, BitsAndBytesConfig
from accelerate import init_empty_weights, infer_auto_device_map

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

def main(pair_path : str, context_path : str, out_path : str, qds : str, ds : str):
    with open(pair_path, 'r') as f:
        text_items = map(lambda x : x.split('\t'), f.readlines())
    qidx, didx = map(list, zip(*text_items))

    with open(context_path, 'r') as f:
        ctx = f.readlines()
    
    # QRELS LOAD
    qdict = None 

    qtext = map(lambda x : qdict[x], qidx)
    # DOC LOOKUP LOAD
    ddict = None 

    dtext = map(lambda x : ddict[x], didx)


    

    with open(out_path, 'w') as f:
        for item in zip(qidx, didx):
            f.write(f'{item[0]}\t{item[1]}\t{item[2]}\n')

    return 0
    
    
if __name__=='__main__':
    fire.Fire(main)