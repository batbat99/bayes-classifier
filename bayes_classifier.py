from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import KernelDensity
import numpy as np
import matplotlib.pyplot as plt

class KDEClassifier(BaseEstimator, ClassifierMixin):
    """Bayesian generative classification based on KDE
    
    Parameters
    ----------
    bandwidth : float
        the kernel bandwidth within each class
    kernel : str
        the kernel name, passed to KernelDensity
    """
    def __init__(self, bandwidth=1.0, kernel='gaussian', n=None, shuffle=False, novelty=0):
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.n = n
        self.shuffle = shuffle
        self.novelty = novelty
        
    def fit(self, X, y):
        X = X[y != 2]
        y = y[y != 2]        
        self.classes_ = np.sort(np.unique(y))
        training_sets = [X[y == yi] for yi in self.classes_]
        if self.novelty:
            self.classes_ = np.hstack((self.classes_, np.array([2])))
        if self.shuffle:
            training_sets = [training_set.sample(frac=1).reset_index(drop=True) for training_set in training_sets]
        if self.n:
            training_sets = [training_set[:self.n] for training_set in training_sets]
        self.models_ = [KernelDensity(bandwidth=self.bandwidth,
                                      kernel=self.kernel).fit(Xi)
                        for Xi in training_sets]
        self.logpriors_ = [np.log(Xi.shape[0] / X.shape[0])
                           for Xi in training_sets]
        return self
        
    def predict_proba(self, X):
        logprobs = np.array([model.score_samples(X)
                             for model in self.models_]).T
        result = np.exp(logprobs + self.logpriors_)
        return result / result.sum(1, keepdims=True)
        
    def predict(self, X):
        proba = self.predict_proba(X)
        print(proba, proba.shape)
        if self.novelty:
            proba = np.hstack((proba, np.ones((proba.shape[0], 1), dtype=proba.dtype) * self.novelty))
        return self.classes_[np.argmax(proba, 1)]

