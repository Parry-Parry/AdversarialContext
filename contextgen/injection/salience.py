from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from fire import Fire
from parryutil import load_yaml
import ir_datasets as irds
from nltk.tokenize import sent_tokenize
import pyterrier as pt 
if not pt.started():
    pt.init()
from pyterrier.io import read_results

def get_salience(config):
    config = load_yaml(config)
    model_id = config['model_id']
    document_file = config['document_file']
    out_file = config['out_file']
    ir_dataset = config['ir_dataset']

    model = SentenceTransformer(model_id)
    documents = read_results(document_file)

    ir_ds = irds.load(ir_dataset)
    docs = pd.DataFrame(ir_ds.docs_iter()).set_index('doc_id').text.to_dict()
    queries = pd.DataFrame(ir_ds.queries_iter()).set_index('query_id').text.to_dict()

    df = []
    for row in documents.itertuples():
        q = queries[row.qid]
        doc = docs[row.docid]
        spans = sent_tokenize(doc)
        sim = cosine_similarity([model.encode(q)], model.encode(spans))
        df.append({'query_id' : row.qid, 'doc_id' : row.docno, 'span' : sim.argmax()})
    
    df = pd.DataFrame.from_records(df).to_csv(out_file, sep='\t', index=False)
    return "Done!"

if __name__ == '__main__':
    Fire(get_salience)

