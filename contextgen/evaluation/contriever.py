from more_itertools import chunked
import numpy as np
import torch
from torch import nn
import ir_datasets
import pyterrier as pt
from transformers import AutoTokenizer, AutoModel


logger = ir_datasets.log.easy()

class BiEncoder(pt.Transformer):

    def __init__(self, model, batch_size=32, text_field='text', verbose=False, tokenizer=None, cuda=None):
        self.model = model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer) if isinstance(tokenizer, str) else tokenizer

        self.cuda = torch.cuda.is_available() if cuda is None else cuda
        if self.cuda:
            self.model = self.model.cuda()
        self.batch_size = batch_size
        self.text_field = text_field
        self.verbose = verbose

    def transform(self, inp):
        columns = set(inp.columns)
        modes = [
            (['qid', 'query', 'docno', self.text_field], self._transform_R),
            (['qid', 'query'], self._transform_Q),
            (['docno', self.text_field], self._transform_D),
        ]
        for fields, fn in modes:
            if all(f in columns for f in fields):
                return fn(inp)
        message = f'Unexpected input with columns: {inp.columns}. Supports:'
        for fields, fn in modes:
            f += f'\n - {fn.__doc__.strip()}: {columns}\n'
        raise RuntimeError(message)

    def _encode_queries(self, texts):
        pass

    def _encode_docs(self, texts):
        pass

    def _transform_D(self, inp):
        """
        Document vectorisation
        """
        res = self._encode_docs(inp[self.text_field])
        return inp.assign(doc_vec=[res[i] for i in range(res.shape[0])])

    def _transform_Q(self, inp):
        """
        Query vectorisation
        """
        it = inp['query']
        if self.verbose:
            it = pt.tqdm(it, desc='Encoding Queries', unit='query')
        res = self._encode_queries(it)
        return inp.assign(query_vec=[res[i] for i in range(res.shape[0])])

    def _transform_R(self, inp):
        """
        Result re-ranking
        """
        return pt.apply.by_query(self._transform_R_byquery, add_ranks=True, verbose=self.verbose)(inp)

    def _transform_R_byquery(self, query_df):
        query_rep = self._encode_queries([query_df['query'].iloc[0]])
        doc_reps = self._encode_docs(query_df[self.text_field])
        scores = (query_rep * doc_reps).sum(axis=1)
        query_df['score'] = scores
        return query_df
    
class ContrieverModel(BiEncoder):
    def __init__(self, model_name="facebook/contriever-msmarco", batch_size : int = 32, **kwargs):
        from src.contriever import Contriever
        print(model_name)
        model = Contriever.from_pretrained(model_name)
        super().__init__(model, tokenizer=model_name, batch_size=batch_size)

    def _encode(self, texts, max_length):
        results = []
        with torch.no_grad():
            for chunk in chunked(texts, self.batch_size):
                inps = self.tokenizer([q for q in chunk], return_tensors='pt', padding=True, truncation=True, max_length=max_length)
                if self.cuda:
                    inps = {k: v.cuda() for k, v in inps.items()}
                res = self.model(**inps)
                results.append(res.cpu().numpy())
        return np.concatenate(results, axis=0)

    def _encode_queries(self, texts):
        return self._encode(texts, max_length=36)
    
    def _encode_docs(self, texts):
        return self._encode(texts, max_length=200)