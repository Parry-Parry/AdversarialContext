from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import numpy as np

def prepare_data(data, encoder = None):
    x, y = data 
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
    X, y, encoder = prepare_data(data) 

    model = LogisticRegression(random_state=42, n_jobs=ncpu)
    model.fit(X, y)
    return model, encoder

def test_regression(data, model, **kwargs):
    from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
    encoder = kwargs.pop('encoder')

    X, y, _ = prepare_data(data, encoder=encoder) 
    pred = model.predict_proba(X)

    pred = np.argmax(pred, axis=-1)

    return {'f1':f1_score(y, pred),'accuracy':accuracy_score(y, pred),'precision':precision_score(y, pred), 'recall':recall_score(y, pred)}