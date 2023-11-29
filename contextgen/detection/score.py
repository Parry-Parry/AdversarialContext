from transformers import AutoModelForSequenceClassification, AutoTokenizer
from fire import Fire 
import torch 
import pandas as pd
import ir_datasets as irds
import pyterrier as pt
if not pt.started():
    pt.init()
from pyterrier.io import read_results

from . import Scorer

def main(model_id : str, 
         in_file : str, 
         out_file : str, 
         window_size : int = 0,
         batch_size : int = 128, 
         trec : bool = False, 
         ir_dataset : str = 'msmarco-passage/trec-dl-2019/judged'):
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = AutoModelForSequenceClassification.from_pretrained(model_id).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    scorer = Scorer(model, tokenizer, window_size, batch_size)

    ds = irds.load(ir_dataset)

    docs = pd.DataFrame(ds.docs_iter()).set_index('doc_id').text.to_dict()
    queries = pd.DataFrame(ds.queries_iter()).set_index('query_id').text.to_dict()

    if trec:
        df = read_results(in_file)
        df['text'] = df['docno'].apply(lambda x : docs[x])
    else:
        df = pd.read_csv(in_file, sep='\t', header=None, names=['qid', 'docno', 'text'], dtype={'qid' : str, 'docno' : str, 'text' : str})
    df['query'] = df['query'].apply(lambda x : queries[x])
    df['score'] = scorer(df['text'].tolist())

    df.to_csv(out_file, sep='\t', index=False, header=False)

    return "Done!"

if __name__ == '__main__':
    Fire(main)


