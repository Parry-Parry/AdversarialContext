import pyterrier as pt
import logging
import re
from collections import Counter, defaultdict
import math
from typing import List, NamedTuple, Tuple, Union
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components

'''
LexRank Implementation using Terrier Index for Corpus Statistics
----------------------------------------------------------------
Allows for the inference of corpus statistics from subset provided aswell as the use of standard terrier indexes
----------------------------------------------------------------
ACKNOWLEDGE:
Sentence Regex from: https://stackoverflow.com/a/31505798
Markov Stationary Distribution Computation partly from https://github.com/crabcamp/lexrank/
'''

alphabets= "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov|edu|me)"
digits = "([0-9])"

def split_into_sentences(text : str) -> List[str]:
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

class LexRanker(pt.Transformer):
    def __init__(self, 
                 setting='summary', 
                 documents=None, 
                 background_index=None, 
                 body_attr='text', 
                 threshold=0., 
                 num_sentences=0, 
                 reverse=False, 
                 norm=True,
                 tokeniser='english', 
                 stemmer='PorterStemmer', 
                 verbose=False) -> None:
        """LexRank Transformer
        ----------------------
        Settings:
            summary -- Returns specified number of sentences ranked by salience in ascending or descending order joined as a string
            sentences -- Returns specified number of sentences ranked by salience in ascending or descending order joined as a list
            ranks -- Returns the sentence idx which would rank the sentences by salience in ascending or descending order as a list
        ----------------------
        Kwargs:
            documents -- Corpus to initialise index
            background_index -- Terrier indexref to intialise index
            body_attr -- Attribute from which to retrieve text for sentence ranking
            threshold -- Threshold for quantisation ~ [0, 1], leave as 0. for no quantisation
            num_sentences -- How many sentences to use in summary, leave as 0 to just rank sentences
            reverse -- If True, take least salient sentences
            norm -- If True, normalise salience scores
            tokeniser -- str name of Terrier tokeniser
            stemmer -- name or Java API reference of Terrier term stemmer 
            verbose -- verbose output
        """
        from pyterrier import autoclass

        self.setting = setting 
        self.indexref = background_index
        self.index = None
        self.body_attr = body_attr
        self.threshold = threshold
        self.num_sentences = num_sentences

        self.reverse = -1 if reverse else 1
        self.norm = norm
        self.verbose = verbose

        self.tokeniser = pt.rewrite.tokenise(tokeniser=tokeniser)
        self.stopwords = autoclass("org.terrier.terms.Stopwords")(None)

        stem_name = f"org.terrier.terms.{stemmer}" if '.' not in stemmer else stemmer
        self.stemmer = autoclass(stem_name)()
        if documents: self.init_index(documents)

        self.outputs = {
            'summary' : self.summary,
            'sentences' : self.list_summary,
            'ranks' : self.ranker
        }
        self.output = self.outputs[setting]
    
    def _text_pipeline(self, text):
        """Tokenise sentences, stem and remove stopwords"""
        tokenised = [sentence.split() for sentence in self.tokeniser(pd.DataFrame({'query':text}))['query'].values]
        stemmed = [self.stemmer.stem(term) for terms in tokenised for term in terms if not self.stopwords.isStopword(term)] 
        return stemmed
    
    def _tf(self, document : NamedTuple) -> Tuple[dict, list]:
        """Split, tokenize and stem sentences then compute term frequencies"""
        sentences = split_into_sentences(getattr(document, self.body_attr)) 
        stemmed = self._text_pipeline(sentences)
        tf = {i : Counter(sentence) for i, sentence in enumerate(stemmed)}

        return tf, sentences
    
    def _idf_cosine(self, i : dict, j : dict, lex, N : int) -> float:
        """Computed IDF modified cosine similarity between two sentences i and j"""
        if i==j: return 1. 
        tokens_i, tokens_j = set(i.keys()), set(j.keys())

        accum = 0
        idf = defaultdict(float)
        for token in tokens_i & tokens_j:
            idf_score = math.log(N / lex[token].getDocumentFrequency())
            idf[token] = idf_score
            accum += i[token] * j[token] * idf_score ** 2
        
        if math.isclose(accum, 0.): return 0.

        mag_i, mag_j = 0, 0

        for token in tokens_i:
            idf_score = idf[token]
            if idf_score == 0.: idf_score = math.log(N / lex[token].getDocumentFrequency())
            tfidf = i[token] * idf_score
            mag_i += tfidf ** 2
        
        for token in tokens_j:
            idf_score = idf[token]
            if idf == 0.: idf_score = math.log(N / lex[token].getDocumentFrequency())
            tfidf = j[token] * idf_score
            mag_j += tfidf ** 2
        
        return accum / math.sqrt(mag_i * mag_j)
    
    def _markov_matrix(self, matrix : np.ndarray) -> np.ndarray:
        """Normalise to create probabilities"""
        if matrix.shape[0] != matrix.shape[1]: raise ValueError('matrix should be square')
        row_sum = matrix.sum(axis=1, keepdims=True)

        return matrix / row_sum

    def _quantized_markov_matrix(self, matrix : np.ndarray) -> np.ndarray:
        """Quantize similarity matrix ~ [0, 1]"""
        _matrix = np.zeros(matrix.shape)
        idx = np.where(_matrix >= self.threshold)
        _matrix[idx] = 1

        return self._markov_matrix(_matrix)
    
    def _connected_nodes(self, matrix : np.ndarray) -> List:
        """Get adjacency matrix"""
        _, labels = connected_components(matrix)
        return [np.where(labels == tag)[0] for tag in np.unique(labels)]

    def _power_method(self, matrix : np.ndarray) -> np.ndarray:
        """Power iteration until convergence"""
        eigenvector = np.ones(matrix.shape[0])
        if eigenvector.shape[0] == 1: return eigenvector

        transition = matrix.T
        while True:
            _next = transition @ eigenvector

            if np.allclose(_next, eigenvector):
                if self.verbose: logging.debug('Converged')
                return _next

            eigenvector = _next
            transition = transition @ transition

    def _stationary_distribution(self, matrix : np.ndarray) -> np.ndarray:
        "Get LexRank Score via eigenvector of transformed similarity matrix"
        distribution = np.zeros(matrix.shape[0])
        grouped_indices = self._connected_nodes(matrix)

        for group in grouped_indices:
            t_matrix = matrix[np.ix_(group, group)]
            eigenvector = self._power_method(t_matrix)
            distribution[group] = eigenvector

        if self.norm:
            distribution /= matrix.shape[0]

        return distribution

    def summary(self, sentences : List[str], scores : np.ndarray) -> str:
        if self.num_sentences != 0: return ' '.join(sentences[np.argsort(scores)[::self.reverse][:self.num_sentences]])
        return ' '.join(sentences[np.argsort(scores)[::self.reverse]])

    def list_summary(self, sentences : List[str], scores : np.ndarray) -> str:
        if self.num_sentences != 0: return sentences[np.argsort(scores)[::self.reverse][:self.num_sentences]]
        return sentences[np.argsort(scores)[::self.reverse]]
    
    def ranker(self, sentences : List[str], scores : np.ndarray) -> List[float]:
        return np.argsort(scores)[::self.reverse].tolist()
        
    def _lexrank(self, doc, lex, N) -> Union[str, List[float], List[str]]:
        logging.debug(f'Computing LexRank for Doc:{doc.docno}')
        # Get sentence level term frequencies
        tf_scores, sentences = self._tf(doc) 
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
        transition = self._quantized_markov_matrix(sim_matrix) if self.threshold !=0. else self._markov_matrix(sim_matrix)
        scores = self._stationary_distribution(transition)

        return self.output(sentences, scores)

    def init_index(self, documents : pd.DataFrame) -> None:
        from pyterrier import DFIndexer, IndexFactory, autoclass
        from pyterrier.index import IndexingType

        indexref = DFIndexer(None, type=IndexingType.MEMORY, verbose=self.verbose).index(documents[self.body_attr], documents["docno"])
        docno2docid = {docno:id for id, docno in enumerate(documents["docno"])} # Keeping this mystery line just in case
        index_docs = IndexFactory.of(indexref)
        docno2docid = {index_docs.getMetaIndex().getItem("docno", i) : i for i in range(index_docs.getCollectionStatistics().getNumberOfDocuments())}
        assert len(docno2docid) == index_docs.getCollectionStatistics().getNumberOfDocuments(), "docno2docid size (%d) doesnt match index (%d)" % (len(docno2docid), index_docs.getCollectionStatistics().getNumberOfDocuments())
        if self.indexref is None:
            self.index = index_docs
        else:
            index_background = IndexFactory.of(self.indexref)
            self.index = autoclass("org.terrier.python.IndexWithBackground")(index_docs, index_background)    

    def transform(self, inp: pd.DataFrame) -> pd.DataFrame:
        assert "docno" in inp.columns and self.body_attr in inp.columns, "Malformed Frame, Expecting Documents"
        documents = inp[["docno", self.body_attr]].drop_duplicates(subset="docno")
        if self.index is None:
             logging.warning('Index not initialized, creating from inputs and a reference if it exists')   
             self.init_index(inp)

        lex = self.index.getLexicon()
        N = self.index.getCollectionStatistics().getNumberOfDocuments()

        documents[self.setting] = documents.apply(lambda x : self._lexrank(x, lex, N), axis=1)
        return documents