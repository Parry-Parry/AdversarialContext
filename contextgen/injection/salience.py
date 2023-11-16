from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from fire import Fire
from parryutil import load_yaml
import ir_datasets as irds
from nltk.tokenize import sent_tokenize

def get_salience(config):
    config = load_yaml(config)
    model_id = config['model_id']
    document_file = config['document_file']
    out_file = config['out_file']
    ir_dataset = config['ir_dataset']

    model = SentenceTransformer(model_id)
    documents = pd.read_csv(document_file, sep='\t', index_col=False)

    ir_ds = irds.load(ir_dataset)
    docs = pd.DataFrame(ir_ds.docs_iter()).set_index('doc_id').text.to_dict()
    queries = pd.DataFrame(ir_ds.queries_iter()).set_index('query_id').text.to_dict()

    queries = {qid : model.encode(q) for qid, q in queries.items()}
    documents = {docid : model.encode(sent_tokenize(doc)) for docid, doc in docs.items()}

    df = []
    for qid, q in queries.items():
        for docid, doc in documents.items():
            sim = cosine_similarity([q], doc)
            df.append({'query_id' : qid, 'doc_id' : docid, 'span' : sim.argmax()})
    
    df = pd.DataFrame.from_records(df).to_csv(out_file, sep='\t', index=False)
    return "Done!"

if __name__ == '__main__':
    Fire(get_salience)

