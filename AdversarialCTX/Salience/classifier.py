from lexrank import LexRank
from lexrank.mappings.stopwords import STOPWORDS
import ir_datasets

class SalienceClassifier:
    def __init__(self, ds, n_text=5, threshold=.1) -> None:
        ds = ir_datasets.load(ds)
        self.docs = [d.text for d in ds.docs_iter()]
        self.lexer = LexRank(self.docs, stopwords=STOPWORDS['en'])
        self.n = n_text 
        self.t = threshold

    def get_summa(self, text):
        groups = text.split('.')
        salient = self.lexer.get_summary(groups, summary_size=self.n, threshold=self.t)
        return salient

    def get_salient(self, text):
        groups = text.split('.')
        scores = self.lexer.rank_sentences(groups, threshold=self.t, fast_power_method=True)
        return groups, scores

    