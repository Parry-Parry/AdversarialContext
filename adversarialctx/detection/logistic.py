from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.corpus import stopwords
import numpy as np

def prepare_data(data):
    x, y = data 
    stop_words = list(stopwords.words('english'))
    encoder = TfidfTransformer(input='content', stop_words=stop_words)

    train_x = encoder.fit_transform(x)
    return train_x, np.array(y), encoder
    
def train_regression(data, **kwargs):
    ncpu = kwargs.pop('ncpu', 1)
    X, y, encoder = prepare_data(data) 

    model = LogisticRegression(random_state=42, n_jobs=ncpu)
    model.fit(X, y)
    return model, encoder

def eval_regression(data, model):
    from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
    X, y = data 
    pred = model.predict_proba(X)
    return {'f1':f1_score(y, pred),'acc':accuracy_score(y, pred),'prec':precision_score(y, pred), 'rec':recall_score(y, pred)}, pred