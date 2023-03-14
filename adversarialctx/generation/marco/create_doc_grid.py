import fire 
import ir_datasets
import pandas as pd

def main(dataset : str, doc_file : str, context_file : str, out_file : str):
    ds = ir_datasets.load(dataset)
    query_lookup = pd.DataFrame(ds.query_iter()).set_index('doc_id').text.to_dict()
    doc_lookup = pd.DataFrame(ds.docs_iter()).set_index('query_id').text.to_dict()

    with open(context_file, 'r') as f:
        context = map(lambda x : x.strip(), f.readlines())

    with open(doc_file, 'r') as f:
        docs = map(lambda x : x.strip().split('\t'), f.readlines())
    qid, did, _ = map(list, zip(*docs))

    with open(out_file, 'w') as f:
        for ctx in context:
            for query, doc in zip(qid, did):
                f.write(f'{ctx}\t{query}\t{doc}\t{query_lookup[query]}\t{doc_lookup[doc]}\n')
    
if __name__ == "__main__":
    fire.Fire(main)