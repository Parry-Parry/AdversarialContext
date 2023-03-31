from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import numpy as np

def prepare_data(data, n_class=2, test=False, encoder = None):
    x, y = data 
    if test: y = list(map(lambda v : np.eye(n_class, dtype=np.float16)[v], y))
    if not encoder:
        stop_words = list(stopwords.words('english'))
        encoder = TfidfVectorizer(input='content', stop_words=stop_words, ngram_range=(1, 2))

        new_x = encoder.fit_transform(x)
        return new_x, np.array(y), encoder
    else:
        new_x = encoder.transform(x)
        return new_x, np.array(y), encoder
    
def train_regression(data, **kwargs):
    ncpu = kwargs.pop('ncpu', 1)
    n_class = kwargs.pop('n_class', 2)
    X, y, encoder = prepare_data(data, n_class) 

    model = LogisticRegression(random_state=42, n_jobs=ncpu)
    model.fit(X, y)
    return model, encoder

def test_regression(data, model, **kwargs):
    from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
    encoder = kwargs.pop('encoder')
    n_class = kwargs.pop('n_class', 2)
    
    X, y, _ = prepare_data(data, n_class, test=True, encoder=encoder) 
    pred = model.predict_proba(X)
    print(pred)
    return {'f1':f1_score(y, pred),'accuracy':accuracy_score(y, pred),'precision':precision_score(y, pred), 'recall':recall_score(y, pred)}