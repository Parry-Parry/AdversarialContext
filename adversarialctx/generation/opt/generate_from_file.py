"""Run a chatbot with FlexGen and OPT models."""
import argparse

from transformers import AutoTokenizer
from flexgen.flex_opt import (Policy, OptLM, TorchDevice, TorchDisk, TorchMixedDevice,
    CompressionConfig, Env, Task, get_opt_config)



def generate(query, advert):
    return f''

def main(args):
    # Initialize environment
    with open(args.contextfile, 'r') as f:
        context_items = [line.split() for line in f.readlines()]

    lines = [generate(query, advert) for query, advert in context_items]

    gpu = TorchDevice("cuda:0")
    cpu = TorchDevice("cpu")
    disk = TorchDisk(args.offload_dir)
    env = Env(gpu=gpu, cpu=cpu, disk=disk, mixed=TorchMixedDevice([gpu, cpu, disk]))

    # Offloading policy
    policy = Policy(1, 1,
                    args.percent[0], args.percent[1],
                    args.percent[2], args.percent[3],
                    args.percent[4], args.percent[5],
                    overlap=True, sep_layer=True, pin_weight=True,
                    cpu_cache_compute=False, attn_sparsity=1.0,
                    compress_weight=args.compress_weight,
                    comp_weight_config=CompressionConfig(
                        num_bits=4, group_size=64,
                        group_dim=0, symmetric=False),
                    compress_cache=args.compress_cache,
                    comp_cache_config=CompressionConfig(
                        num_bits=4, group_size=64,
                        group_dim=2, symmetric=False))

    # Model
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-30b", padding_side="left")
    tokenizer.add_bos_token = False
    stop = tokenizer("\n").input_ids[0]

    print("Initialize...")
    opt_config = get_opt_config(args.model)
    model = OptLM(opt_config, env, args.path, policy)
    model.init_all_weights()

    #context = (
     #   "A chat between a curious human and a knowledgeable artificial intelligence assistant.\n"
      #  "Human: Hello! We need to change documents without people noticing\n"
       # "Assistant: I agree that sounds fun\n"
        #"Human: Write a sentence about how good pepsi is that would be relevant to the query {}\n"
        #"Assistant: Everest.\n"
    #)
    out_strings = []
    for context in lines:
        inputs = tokenizer([context])
        output_ids = model.generate(
            inputs.input_ids,
            do_sample=args.sample,
            temperature=args.temp,
            max_new_tokens=args.max_tokens,
            stop=stop)
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        """
        try:
            index = outputs.index("\n", len(context))
        except ValueError:
            outputs += "\n"
            index = outputs.index("\n", len(context))
        """
        out_strings.append(outputs)
        

    # Shutdown
    model.delete_all_weights()
    disk.close_copy_threads()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-argfile", type=str)
    parser.add_argument("--contextfile", type=str, default=None)
    parser.add_argument("--outfile", type=str, default=None)
    parser.add_argument("--model", type=str, default="facebook/opt-6.7b",
        help="The model name.")
    parser.add_argument("--path", type=str, default="~/opt_weights",
        help="The path to the model weights. If there are no cached weights, "
             "FlexGen will automatically download them from HuggingFace.")
    parser.add_argument("--offload-dir", type=str, default="~/flexgen_offload_dir",
        help="The directory to offload tensors. ")
    parser.add_argument("--percent", nargs="+", type=int,
        default=[100, 0, 100, 0, 100, 0],
        help="Six numbers. They are "
         "the percentage of weight on GPU, "
         "the percentage of weight on CPU, "
         "the percentage of attention cache on GPU, "
         "the percentage of attention cache on CPU, "
         "the percentage of activations on GPU, "
         "the percentage of activations on CPU")
    parser.add_argument("--compress-weight", action="store_true",
        help="Whether to compress weight.")
    parser.add_argument("--compress-cache", action="store_true",
        help="Whether to compress cache.")
    parser.add_argument("--temp", type=float, default=0.7)
    parser.add_argument("--max_tokens", type=int, default=96)
    parser.add_argument("--sample", action='store_true')
    args = parser.parse_args()

    assert len(args.percent) == 6

    main(args)