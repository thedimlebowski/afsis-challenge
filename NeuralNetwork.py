import numpy as np
import random as rand
from ffnet import ffnet, mlgraph
import networkx as NX
from joblib import Parallel, delayed
from copy import copy

def train_once(net, train_input, train_output, in_cv, test_input, test_output, i):
    net.randomweights()
    net.train_cg(train_input, train_output, disp = 0)
    train_error = net.sqerror(train_input,train_output)
    if in_cv:
        test_error = net.sqerror(test_input,test_output)
    else:
        test_error = None
    if net.verbose:
        if in_cv:
            print 'Optimisation {0}: train-error {1:.2e}, test-error {2:.2e}'.format(i,train_error,test_error)
        else:
            print 'Optimisation {0}: train-error {1:.2e}'.format(i,train_error)
    return train_error, test_error, net

def cross_validation(net, input, output, ratio, ntry, i, nproc):
    (test_input, train_input, test_output, train_output) = net.random_split(input, output, ratio)
    net.train_best(train_input = train_input, train_output = train_output, ntry = ntry, in_cv = True, test_input = test_input, test_output = test_output, nproc = nproc)
    train_error = net.sqerror(train_input, train_output)
    test_error = net.sqerror(test_input, test_output)
    if net.verbose:
        print 'Cross-validation {0}: train-error {1:.2e}, test-error {2:.2e}'.format(i,train_error,test_error)
    return net

class NeuralNetwork(ffnet):

    def __init__(self, arch, verbose = False):
        """
        Initialises the neural network
        """
        self.verbose = verbose
        self.arch = arch
        conec = mlgraph(self.arch, biases=True)
        ffnet.__init__(self, conec)

    def train_cv(self, input, output, ratio = 0.75, ncv = 10, ntry = 10, nproc = 1):
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
                                delayed(cross_validation)(self, input, output, ratio, ntry, i, nproc)
                                    for i in range(1,ntry+1))

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

    def train_best(self, train_input, train_output, ntry = 10, in_cv = False, test_input = None, test_output = None, nproc = 1):
        """
        Finds best neural network amongst 10 trained neural networks
        The training method is a conjugate gradient descent
        The fitness is measured by sqerror
        """
        results =   Parallel(n_jobs = nproc)(
                        delayed(train_once)(self, train_input, train_output, in_cv, test_input, test_output, i)
                            for i in range(1,ntry+1))
        train_error = [result[0] for result in results]
        test_error = [result[1] for result in results]
        net = [result[2] for result in results]
        self.best_net = net[np.argmin(train_error)]
        self.__dict__.update(self.best_net.__dict__)

    def sim(self, input, cv = False, best = False, cv_i = False):
        """
        Simulates the result of the neural network obtained by either training method
        """
        if cv:
            output_list = []
            for i in range(self.ncv):
                self.__dict__.update(self.cv_nets[i].__dict__)
                output_list += [self.call(input)]
            output = np.concatenate(output_list, axis = 1)
            mean_output = np.mean(output,1)
            nsample = np.shape(input)[0]
            return np.resize(mean_output,(nsample,self.arch[-1]))
        if best:
            self.__dict__.update(self.best_net.__dict__)
            return self.call(input)
        if cv_i:
            self.__dict__.update(self.cv_nets[cv_i].__dict__)
            return self.call(input)
        return self.call(input)
