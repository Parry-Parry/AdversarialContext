from fire import Fire 
import pandas as pd
import pyterrier as pt
if not pt.started():
    pt.init()
from pyterrier.io import read_results

def construct(item_file : str, document_file : str, out_file : str):
    with open(item_file, 'r') as f: 
        lines = [*map(lambda x : x.strip(), f.readlines())]
        item_spans = [*map(lambda x : x.split('\t'), lines)]
    
    documents = read_results(document_file)

    df = []

    for item, span in item_spans:
        for row in documents.itertuples():
            df.append({
                'qid' : row.qid,
                'docno' : row.docno,
                'span' : span,
                'item' : item
            })
    
    pd.DataFrame.from_records(df).to_csv(out_file, sep='\t', index=False)
    
    return "Done!"

if __name__ == '__main__':
    Fire(construct)