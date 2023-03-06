import fire 
import ir_datasets

def main(dataset : str, in_file : str, out_file : str):
    ds = ir_datasets.load(dataset)
    queries = list(ds.queries_iter())
    with open(in_file, 'r') as f:
        context = f.readlines()
    with open(out_file, 'w') as f:
        for ctx in context:
            for query in queries:
                f.write(f'{ctx}\t{query.query_id}\t{query.text}')
    
if __name__ == "__main__":
    fire.Fire(main)