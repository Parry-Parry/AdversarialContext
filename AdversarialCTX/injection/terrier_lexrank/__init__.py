import pyterrier as pt
import logging
import re
from collections import Counter
import math
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components

'''
LexRank Implementation using Terrier Index for Corpus Statistics
----------------------------------------------------------------
Implementation basically just removes the need to compute IDF over massive corpora that have an index in Terrier

Sentence Regex from: https://stackoverflow.com/a/31505798
Markov Stationary Distribution Computation from https://github.com/crabcamp/lexrank/
'''

alphabets= "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov|edu|me)"
digits = "([0-9])"

def split_into_sentences(text):
    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    text = re.sub(digits + "[.]" + digits,"\\1<prd>\\2",text)
    if "..." in text: text = text.replace("...","<prd><prd><prd>")
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    return sentences

class LexRanker:
    def __init__(self, background_index=None, body_attr='text', threshold=.1, norm=False, verbose=False) -> None:
        from pyterrier import autoclass

        self.indexref = background_index if background_index else None
        self.body_attr = body_attr
        self.threshold = threshold
        self.norm = norm
        self.verbose = verbose

        # Will change this to use terrier enum interface once I understand how to do custom termpipelines
        self.tokenizer = pt.rewrite.tokenise()
        self.stopwords = autoclass("org.terrier.terms.Stopwords")
        self.stemmer = autoclass("org.terrier.terms.PorterStemmer")
        
    def _tokenize(self, text):
        return self.tokenizer.tokenise(text)
    
    def _stem(self, terms):
        return [self.stemmer.stem(term) for term in terms if not self.stopwords.isStopword(term)] 

    def _tf(self, document):
        scores = {}
        sentences = split_into_sentences(getattr(document, self.body_attr)) 
        tokenized = [self.tokenize(sentence) for sentence in sentences] 
        stemmed = [self.stem(terms) for terms in tokenized]
        for i, sentence in enumerate(stemmed):
            scores[i] = Counter(sentence)
        return scores
    
    def _idf_cosine(i, j, lex, N):
        if i==j: return 1. 
        tokens_i, tokens_j = set(i.keys()), set(j.keys())

        accum = 0
        idf = {}
        for token in tokens_i & tokens_j:
            idf_score = math.log(N / lex[token].getDocumentFrequency() + 1e-20)
            idf[token] = idf_score
            accum += i[token] * j[token] * idf_score ** 2
        
        if math.isclose(accum, 0): return 0

        mag_i, mag_j = 0, 0

        for token in tokens_i:
            tfidf = i[token] * idf[token]
            mag_i += tfidf ** 2
        
        for token in tokens_j:
            tfidf = j[token] * idf[token]
            mag_j += tfidf ** 2
        
        return accum / math.sqrt(mag_i * mag_j)
    
    def _markov_matrix(matrix):
        if matrix.shape[0] != matrix.shape[1]: raise ValueError('matrix should be square')
        row_sum = matrix.sum(axis=1, keepdims=True)

        return matrix / row_sum

    def _quantized_markov_matrix(self, matrix):
        _matrix = np.zeros(matrix.shape)
        idx = np.where(_matrix >= self.threshold)
        _matrix[idx] = 1

        return self._markov_matrix(_matrix)
    
    def _connected_nodes(self, matrix):
        _, labels = connected_components(matrix)

        groups = []

        for tag in np.unique(labels):
            group = np.where(labels == tag)[0]
            groups.append(group)

        return groups


    def _power_method(self, matrix):
        eigenvector = np.ones(len(matrix))

        if len(eigenvector) == 1:
            return eigenvector

        transition = matrix.transpose()

        while True:
            eigenvector_next = np.dot(transition, eigenvector)

            if np.allclose(eigenvector_next, eigenvector):
                if self.verbose: logging.info('Converged')
                return eigenvector_next

            eigenvector = eigenvector_next
    
            transition = np.dot(transition, transition)

    
    def stationary_distribution(self, matrix):
        distribution = np.zeros(matrix.shape[0])
        grouped_indices = self._connected_nodes(matrix)

        for group in grouped_indices:
            t_matrix = matrix[np.ix_(group, group)]
            eigenvector = self._power_method(t_matrix)
            distribution[group] = eigenvector

        if self.norm:
            distribution /= matrix.shape[0]

        return distribution
        
    def _lexrank(self, doc, lex, N):
        tf_scores = self._tf(doc) # Get sentence level frequencies
        dim = len(tf_scores)
        
        # Construct similarity matrix
        sim_matrix = np.zeros((dim, dim))
        for i in range(dim):
            for j in range(dim):
                sim = self._idf_cosine(tf_scores[i], tf_scores[j], lex, N)
                if sim:
                    sim_matrix[i, j] = sim
                    sim_matrix[j, i] = sim
        
        # Compute Stationary Distribution 
        transition = self._quantized_markov_matrix(sim_matrix) if self.threshold is not None else self._markov_matrix(sim_matrix)
        scores = self.stationary_distribution(transition)
        return np.argsort(scores).tolist()

    def transform(self, inp):
        from pyterrier import DFIndexer, IndexFactory, autoclass
        from pyterrier.index import IndexingType

        # General Text Scorer Boilerplate from pt.TextIndexProcessor
        documents = inp[["docno", self.body_attr]].drop_duplicates(subset="docno")
        indexref = DFIndexer(None, type=IndexingType.MEMORY, verbose=self.verbose).index(documents[self.body_attr], documents["docno"])
        docno2docid = { docno:id for id, docno in enumerate(documents["docno"]) } # Keeping this mystery line just in case
        index_docs = IndexFactory.of(indexref)
        docno2docid = {index_docs.getMetaIndex().getItem("docno", i) : i for i in range(index_docs.getCollectionStatistics().getNumberOfDocuments())}
        assert len(docno2docid) == index_docs.getCollectionStatistics().getNumberOfDocuments(), "docno2docid size (%d) doesnt match index (%d)" % (len(docno2docid), index_docs.getCollectionStatistics().getNumberOfDocuments())
        if self.indexref is None:
            index = index_docs
        else:
            index_background = IndexFactory.of(self.indexref)
            index = autoclass("org.terrier.python.IndexWithBackground")(index_docs, index_background)          
        
        res = []
        build_record = lambda d, r : {'docno':d, 'ranks':r}

        ## LexRank ##

        lex = index.getLexicon()
        N = index.getCollectionStatistics().getNumberOfDocuments()

        for doc in documents.itertuples():
            if self.verbose: logging.info(f'Computing LexRank for Doc:{doc.docno}')
            ranks = self._lexrank(doc, lex, N)
            res.append(build_record(doc.docno, ranks))

        return pd.DataFrame.from_records(res)