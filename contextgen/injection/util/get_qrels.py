from fire import Fire 
import pandas as pd
import ir_datasets as irds
from nltk.tokenize import sent_tokenize
import numpy.random as random
import pyterrier as pt
if not pt.started():
    pt.init()
from pyterrier.io import read_results


random.seed(42)

rel_lookup = {
    'relevant' : [2, 3],
    'non-relevant' : [0, 1],
}

def get_qrels(document_file : str, 
              eval_set : str, 
              out_file : str, 
              rel : str = 'relevant'):
    documents = read_results(document_file)
    ir_dataset = irds.load(eval_set)
    docs = pd.DataFrame(ir_dataset.docs_iter()).set_index('doc_id').text.to_dict()
    qrels = pd.DataFrame(ir_dataset.qrels_iter())
    if rel != 'irrelevant':
        qrels = qrels[qrels.relevance.isin(rel_lookup[rel])]
        qrels = qrels.groupby('query_id').doc_id.apply(list).to_dict()
    else:
        qrels = qrels[qrels.relevance > 0]

    df = []

    for row in documents.itertuples():
        docno = str(row.docno)
        qid = str(row.qid)
        
        if rel == 'irrelevant':
            qrel_docs = [d for q, d in qrels.items() if q != qid]
            qrel_docs = [d for doc in qrel_docs for d in doc]
        else:
            qrel_docs = qrels[qid]
        random_doc = random.choice(qrel_docs)
        random_span = random.choice(sent_tokenize(docs[random_doc]))

        df.append({'qid' : qid, 'docno' : docno, 'span' : random_span})
    
    df = pd.DataFrame.from_records(df)
    df.to_csv(out_file, sep='\t', index=False)

    return "Done!"

if __name__ == '__main__':
    Fire(get_qrels)
        
    


    