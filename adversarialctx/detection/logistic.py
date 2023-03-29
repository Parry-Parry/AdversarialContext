from sklearn.linear_model import LogisticRegression
def train_regression(data, **kwargs):
    ncpu = kwargs.pop('ncpu', 1)
    X, y, test_X, test_y = data 

    model = LogisticRegression(random_state=42, n_jobs=ncpu)
    model.fit(X, y)