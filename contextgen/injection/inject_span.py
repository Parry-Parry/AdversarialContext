from fire import Fire
import pandas as pd
from parryutil import load_yaml
import ir_datasets as irds
 
from . import SalientSyringe, AbsoluteSyringe

def do_span(document_file : str, span_file : str, out_file : str, type : str, ir_dataset : str, salience_file : str = None, item_file : str = None):
    pos = type.split('_')[1]

    if 'absolute' in type: syringe = AbsoluteSyringe(pos=pos)
    else: syringe = SalientSyringe(pos=pos, salience_file=salience_file)

    documents = pd.read_csv(document_file, sep='\t', index_col=False)
    spans = pd.read_csv(span_file, sep='\t', index_col=False)

    ir_ds = irds.load(ir_dataset)
    docs = pd.DataFrame(ir_ds.docs_iter()).set_index('doc_id').text.to_dict()

    with open(item_file, 'r') as f: items = [*map(lambda x : x.strip(), f.readlines())]

    df = []
    if items:
        df = []
        out_file = out_file.replace('.tsv.gz', '')
        for item in items:
            span_subset = spans[spans.item == item].set_index('docno').span.to_dict()
            for row in documents:
                docno = row.docno
                qid = row.qid
                text = docs[docno]
                span = span_subset[docno]
                text = syringe(text, span, qid=qid, docno=docno)
                df.append({'qid' : qid, 'docno' : docno, 'text' : text, 'item' : item})
        pd.DataFrame.from_records(df).to_csv(f'{out_file}.{item}.tsv.gz', sep='\t', index=False)
    else:
        span_subset = spans.set_index(['qid', 'docno']).span.to_dict()
        for row in documents:
            docno = row.docno
            qid = row.qid
            text = docs[docno]
            span = span_subset[(qid, docno)]
            text = syringe(text, span, qid=qid, docno=docno)
            df.append({'qid' : qid, 'docno' : docno, 'text' : text, 'item' : 'NA'})
        pd.DataFrame.from_records(df).to_csv(out_file, sep='\t', index=False)
    return "Done!"

if __name__ == '__main__':
    Fire(do_span)

