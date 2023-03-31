from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import numpy as np

def prepare_data(data):
    x, y = data 
    stop_words = list(stopwords.words('english'))
    encoder = TfidfVectorizer(input='content', stop_words=stop_words, ngram_range=(1, 2))

    train_x = encoder.fit_transform(x)
    return train_x, np.array(y), encoder
    
def train_regression(data, **kwargs):
    ncpu = kwargs.pop('ncpu', 1)
    X, y, _ = prepare_data(data) 

    model = LogisticRegression(random_state=42, n_jobs=ncpu)
    model.fit(X, y)
    return model

def test_regression(data, model, **kwargs):
    from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
    X, y, _ = prepare_data(data) 
    pred = model.predict_proba(X)
    return {'f1':f1_score(y, pred),'accuracy':accuracy_score(y, pred),'precision':precision_score(y, pred), 'recall':recall_score(y, pred)}