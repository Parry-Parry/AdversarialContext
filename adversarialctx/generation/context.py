import fire

def main(context_path : str, sentence_path : str, out_path : str):
    with open(context_path, 'r') as f:
        context = f.readlines()
    with open(sentence_path, 'r') as f:
        sentences = f.readlines()

    with open(out_path, 'w') as f:
        for item in zip(context, sentences):
            f.write(f'{item[0]}\t{item[1]}\n')

    return 0
    
if __name__=='__main__':
    fire.Fire(main)