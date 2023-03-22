import fire

def format_string(string):
    string = string.strip()
    sentences = string.split('.')[::-1]
    return '.'.join(sentences)

def main(file_path : str, out_path : str):
    with open(file_path, 'r') as f:
        text_items = map(lambda x : x.split('\t'), f.readlines())

    qidx, didx, sx = map(list, zip(*text_items))

    nsx = map(lambda x : format_string(x), sx)

    with open(out_path, 'w') as f:
        for item in zip(qidx, didx, nsx):
            f.write(f'{item[0]}\t{item[1]}\t{item[2]}\n')

    return 0
    
    
if __name__=='__main__':
    fire.Fire(main)