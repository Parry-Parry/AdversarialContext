import pyterrier as pt
if not pt.started():
    pt.init()
from fire import Fire 
from pyterrier.io import write_results
from pyterrier_pisa import PisaIndex
import pandas as pd
import ir_datasets as irds

def get_bm25(index : str, dataset : str, out_file : str, cutoff : int = 100):
    index = PisaIndex.from_dataset(index, threads=4)
    bm25 = index.bm25(num_results = cutoff)

    ds = irds.load(dataset)
    queries = pd.DataFrame(ds.queries_iter()).rename(columns={'query_id' : 'qid', 'text' : 'query'})
    result = bm25.transform(queries)

    write_results(result, out_file)

    return "Done!"

if __name__ == '__main__':
    Fire(get_bm25)


