import pyterrier as pt
pt.init()

class LexRanker(pt.TextIndexProcessor):
    def __init__(self, index) -> None:
        self.index = index 
        self.lexicon = index.getLexicon()
    def transform(self, docs):
        pass