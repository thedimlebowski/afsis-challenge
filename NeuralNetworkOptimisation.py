import numpy as np
import random as rand
from ffnet import ffnet, mlgraph
import networkx as NX
from joblib import Parallel, delayed
from copy import copy, deepcopy
import sys
import matplotlib.pyplot as plt


def train_once(nno, train_input, train_output, in_cv, test_input, test_output, i):
    nno = deepcopy(nno)
    nno.net.randomweights()
    nsample = np.shape(train_input)[0]
    order = rand.sample(range(nsample),nsample)
    nno.net.train_cg(train_input[order,:], train_output[order,:], disp = 0)
    train_error = nno.net.sqerror(train_input,train_output)
    if in_cv:
        test_error = nno.net.sqerror(test_input,test_output)
    else:
        test_error = None
    if nno.verbose:
        if in_cv:
            print '\nOptimisation {0}: train-error {1:.2e}, test-error {2:.2e}'.format(i,train_error,test_error)
            sys.stdout.flush()
        else:
            print '\nOptimisation {0}: train-error {1:.2e}'.format(i,train_error)
            sys.stdout.flush()
    return train_error, test_error, nno.net

def cross_validation(nno, input, output, ratio, ntry, i, nproc, best_test):

    nno = deepcopy(nno)
    (test_input, train_input, test_output, train_output) = nno.random_split(input, output, ratio)
    nno.train_best(train_input = train_input, train_output = train_output, ntry = ntry, in_cv = True, test_input = test_input, test_output = test_output, nproc = nproc, best_test = best_test)
    train_error = nno.best_net.sqerror(train_input, train_output)
    test_error = nno.best_net.sqerror(test_input, test_output)
    if nno.verbose:
        print '\nCross-validation {0}: train-error {1:.2e}, test-error {2:.2e}'.format(i,train_error,test_error)
        sys.stdout.flush()
    return nno.best_net

class NeuralNetworkOptimisation(object):
    def __init__(self, arch, verbose = False, biases = True):
        """
        Initialises the neural network
        """
        self.verbose = verbose
        self.arch = arch
        self.conec = mlgraph(self.arch, biases = biases)
        self.net = ffnet(self.conec)

    def train_best(self, train_input, train_output, ntry = 10, in_cv = False, test_input = None, test_output = None, nproc = 1, best_test = False):
        """
        Finds best neural network amongst 10 trained neural networks
        The training method is a conjugate gradient descent
        The fitness is measured by sqerror
        """

        results =   Parallel(n_jobs = nproc, backend="threading")(
                        delayed(train_once)(self, train_input, train_output, in_cv, test_input, test_output, i)
                            for i in range(1,ntry+1))
        train_error = [result[0] for result in results]
        test_error = [result[1] for result in results]
        nets = [result[2] for result in results]
        self.best_net = nets[np.argmin(train_error)]
        if best_test == True:
            self.best_net = nets[np.argmin(test_error)]
        print '\nOptimisation done. train-error {0:.2e}'.format(np.min(train_error))

    def train_cv(self, input, output, ratio = 0.75, ncv = 10, ntry = 10, nproc = 1, best_test = False):
        """
        Train the network with a cross-validation strategy
        Splits the input in a train and test sets with ratio = len(train)/len(input)
        """
        self.ncv = ncv
        if ratio == 1:
            print 'r=1. Use train_best instead...'
            self.train_best(input, output, ntry)
        else:
            self.cv_nets =   Parallel(n_jobs = nproc)(
                                delayed(cross_validation)(self, input, output, ratio, ntry, i, nproc, best_test)
                                    for i in range(1,ncv+1))


    def random_split(self, input, output, ratio = 0.75):
            """
            Splits input and output arrays in two datasets each (test and train) according to the ratio specified
            """
            nsample = np.shape(input)[0]
            assert nsample == np.shape(output)[0]

            ntrain = int(ratio*nsample)

            train_indices = rand.sample(range(nsample),ntrain)
            test_indices = list(set(range(nsample)) - set(train_indices))

            input_test = input[test_indices,:]
            input_train = input[train_indices,:]

            output_test = output[test_indices,:]
            output_train = output[train_indices,:]
            return (input_test, input_train, output_test, output_train)

    def sim(self, input, cv = False, best = False, cv_i = False):
        """
        Simulates the result of the neural network obtained by either training method
        """
        if cv:
            nsample = np.shape(input)[0]
            noutput = self.arch[-1]
            output = np.zeros((nsample,noutput))
            for i in range(self.ncv):
                output += self.cv_nets[i].call(input)/(self.ncv)
            return output
        if best:
            return self.best_net.call(input)
        if type(cv_i) == int:
            return self.cv_nets[cv_i].call(input)
        else:
            return self.net.call(input)

    def plot(self):
        NX.draw_graphviz(self.net.graph, prog = 'dot')
        plt.show()
