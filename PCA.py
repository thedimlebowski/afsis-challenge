import numpy as np
from copy import copy

class PCA(object):

    def __init__(self, nred):
        self.nred = nred

    def _normalise(self, X, type):
        if type == 'fit':
            self.mean = np.mean(X,0)
            self.std = np.std(X,0)
        X_norm = (X - self.mean)/self.std
        return X_norm

    def fit(self, X_in, normalise = True):
        n_sample = np.shape(X_in)[0]
        n_dof = np.shape(X_in)[1]
        if self.nred > n_dof:
            raise StandardError('self.nred > n_dof')
        print 'number of dof: {0}'.format(n_dof)
        print 'number of samples: {0}'.format(n_sample)
        if normalise:
            X_norm = self._normalise(X_in, 'fit')
            X = copy(X_norm)
        else:
            X = copy(X_in)
        (U,s,V) = np.linalg.svd(X)
        S = np.zeros(np.shape(X))
        S[:self.nred,:self.nred] = np.diag(s[:self.nred])
        accuracy = 100*(1-np.linalg.norm(X-np.dot(np.dot(U,S),V))/np.linalg.norm(X))
        print 'Data appromixated up to {0:.2f}% with {1} modes.'.format(accuracy, self.nred)
        self.basis = V[:self.nred,:]

    def predict(self, X_in, normalise = True):
        n_dof = np.shape(X_in)[1]
        if self.nred > n_dof:
            raise StandardError('self.nred > n_dof')
        if normalise:
            X_norm = self._normalise(X_in, 'predict')
            X = copy(X_norm)
        else:
            X = copy(X_in)
        return np.dot(X,self.basis.T)





