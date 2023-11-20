from typing import Any
import pandas as pd
from nltk.tokenize import sent_tokenize
import numpy.random as random

random.seed(42)

class SalientSyringe(object):
    def __init__(self, pos : str, salience_file : str) -> None:
        if pos == 'before': self.inject = self.before
        elif pos == 'after': self.inject = self.after
        else: raise ValueError(f"Invalid position {pos}")

        salience = pd.read_csv(salience_file, sep='\t', index_col=False, dtype={'qid' : str, 'doc_id' : str, 'span' : int})
        self.salience = salience.set_index(['query_id', 'doc_id']).span.to_dict()

    def before(self, spans, span, i) -> str:
        if i == 0: return ' '.join([span, *spans])
        return ' '.join([*spans[:i], span, *spans[i:]])
    
    def after(self, spans, span, i) -> str:
        if i == len(spans) - 1: return ' '.join([*spans, span])
        return ' '.join([*spans[:i+1], span, *spans[i+1:]])

    def __call__(self, text, span, docno=None, qid=None) -> str:
        spans = sent_tokenize(text)
        salience = self.salience[(str(qid), str(docno))]
        return self.inject(spans, span, salience)

class AbsoluteSyringe(object):
    def __init__(self, pos : str) -> None:
        if pos == 'before': self.inject = self.before
        elif pos == 'middle': self.inject = self.middle
        elif pos == 'after': self.inject = self.after
        else: raise ValueError(f"Invalid position {pos}")

    def before(self, spans, span) -> str:
        return ' '.join([span, *spans])
    
    def middle(self, spans, span) -> str:
        if len(spans) == 1:
            if random.random() > 0.5: return self.before(spans, span)
            else: return self.after(spans, span)
        else:
            split = len(spans) // 2
            return ' '.join([*spans[:split], span, *spans[split:]])

    def after(self, spans, span) -> str:
        new = [*spans, span]
        return ' '.join(new)
    
    def __call__(self, text, span, **kwargs) -> str:
        return self.inject(sent_tokenize(text), span)