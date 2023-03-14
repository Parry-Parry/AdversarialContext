import fire 
import ir_datasets
import pandas as pd

from pyterrier_freetext.summary.ranker import SentenceRanker

def main(summary_model : str, docs : str, queries: str, doc_file : str, context_file : str, out_file : str):
    doc_lookup = pd.DataFrame(ir_datasets.load(docs).docs_iter()).set_index('doc_id').text.to_dict()
    query_lookup = pd.DataFrame(ir_datasets.load(queries).queries_iter()).set_index('query_id').text.to_dict()

    with open(context_file, 'r') as f:
        context = map(lambda x : x.strip(), f.readlines())

    with open(doc_file, 'r') as f:
        docs = map(lambda x : x.strip().split('\t'), f.readlines())
    qid, did, _ = map(list, zip(*docs))

    qtexts = [query_lookup[q] for q in qid]
    dtexts = [doc_lookup[d] for d in did]

    ranker = SentenceRanker(summary_model, mode='summary', num_sentences=1, out_attr='sentence')
    output = ranker.transform(pd.DataFrame({'query':qtexts, 'text':dtexts}))['sentence'].tolist()

    with open(out_file, 'w') as f:
        for ctx in context:
            for query, doc, qtext, out in zip(qid, did, qtexts, output):
                f.write(f'{ctx}\t{query}\t{doc}\t{qtext}\t{out}\n')
    
if __name__ == "__main__":
    fire.Fire(main)