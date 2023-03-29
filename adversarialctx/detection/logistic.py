from sklearn.linear_model import LogisticRegression
def train_regression(data, **kwargs):
    ncpu = kwargs.pop('ncpu', 1)
    X, y = data 

    model = LogisticRegression(random_state=42, n_jobs=ncpu)
    model.fit(X, y)
    return model

def eval_regression(data, model):
    X, y = data 
    pred = model.predict_proba(X)