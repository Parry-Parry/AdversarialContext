import re
import pyterrier as pt
import pandas as pd
if not pt.started():
    pt.init()

### MODELS ###

def clean_text(text):
    text = re.sub(r'[^A-Za-z0-9 ]+', '', text)
    return re.sub(r'/[^\x00-\x7F]/g', '', text).strip()

def clean(df):
    df['text'] = df['text'].apply(lambda x : clean_text(x))
    df['query'] = df['query'].apply(lambda x : clean_text(x))
    return df

def load_colbert(**kwargs):
    checkpoint = kwargs['checkpoint']
    device = kwargs.get('device', True)
    from pyterrier_colbert.ranking import ColBERTModelOnlyFactory
    pytcolbert = ColBERTModelOnlyFactory(checkpoint, gpu=device)
    return pytcolbert.text_scorer()

def load_tasb(**kwargs):
    checkpoint = kwargs['checkpoint']
    from pyterrier_dr import TasB, BiScorer
    return BiScorer(TasB(checkpoint))

def load_t5(**kwargs):
    from pyterrier_t5 import MonoT5ReRanker
    return MonoT5ReRanker()

def load_bm25(**kwargs):
    dataset = kwargs['dataset']
    variant = kwargs.get('variant', 'terrier_stemmed')
    ds = pt.get_dataset(dataset)
    indx = pt.IndexFactory.of(ds.get_index(variant=variant), memory=True)
    scorer = pt.apply.generic(lambda x : clean(x)) >> pt.batchretrieve.TextScorer(body_attr='text', wmodel='BM25', background_index=indx, properties={"termpipelines" : "Stopwords,PorterStemmer"})
    return scorer

def load_model(**kwargs):
    if kwargs['model'] == 'bm25': return load_bm25(**kwargs)
    if kwargs['model'] == 'tasb': return load_tasb(**kwargs)
    if kwargs['model'] == 't5': return load_t5(**kwargs)
    if kwargs['model'] == 'colbert': return load_colbert(**kwargs)
    raise ValueError(f"Unknown model {kwargs['model']}")

### METRICS ###

def build_rank_lookup(df):
    frame = {}
    for qid in df.qid.unique().tolist():
        sub = df[df.qid==qid].copy()
        assert len(sub) > 0, f"Query {qid} has no results"
        frame[qid] = [(row.docno, row.score) for row in sub.itertuples()]
    return frame

def ABNIRML(score, adv_score):
    diff = score - adv_score
    if diff < 0: return -1 
    elif diff > 0: return 1
    return 0

def get_ranks(docno, score, lookup):
    old_ranks = [(k, v) for k, v in lookup]
    old_ranks.sort(key=lambda x : x[1], reverse=True)
    old_rank = [i for i, item in enumerate(old_ranks) if item[0]==docno]
    new_ranks = [item for item in old_ranks if item[0] != docno]
    new_ranks.append((docno, score))
    new_ranks.sort(reverse=True, key=lambda x : x[1])
    return old_rank[0], [i for i, item in enumerate(new_ranks) if item[0]==docno][0]

def MRC(docno, score, lookup):
    old_ranks = [(k, v) for k, v in lookup]
    old_ranks.sort(key=lambda x : x[1], reverse=True)
    old_rank = [i for i, item in enumerate(old_ranks) if item[0]==docno]
    new_ranks = [item for item in old_ranks if item[0] != docno]
    new_ranks.append((docno, score))
    new_ranks.sort(reverse=True, key=lambda x : x[1])
    rank_change = old_rank[0] - [i for i, item in enumerate(new_ranks) if item[0]==docno][0]
    return rank_change

def add_new_ids(df, max_id, qrels):
    new = []
    new_ids = {}
    for i, row in enumerate(df.itertuples()):
        sub = qrels.loc[qrels.doc_id == row.doc_id].copy()
        docid = str(max_id + i)
        new_ids[row.doc_id] = docid
        for subrow in sub.itertuples():
            new.append({
                'query_id' : subrow.query_id,
                'doc_id' : docid,
                'relevance' : subrow.relevance,
                'iteration' : subrow.iteration
            })
    new_qrels = pd.DataFrame.from_records(new)
    return new_ids, new_qrels