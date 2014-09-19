import numpy as np
import random as rand
from ffnet import ffnet, mlgraph
import networkx as NX
from joblib import Parallel, delayed
from copy import copy, deepcopy


# This functions are outside the class to make the parallel implementation possible
def cross_validation( net, input, output, ratio, ntry, iteration, nproc = 1):
    (test_input, train_input, test_output, train_output) = net.random_split(input, output, ratio)
    net.train_best(train_input = train_input, train_output = train_output, ntry = ntry, in_cv = True, test_input = test_input, test_output = test_output, nproc = nproc)
    cv_weights = net.weights
    train_error = net.sqerror(train_input, train_output)
    test_error = net.sqerror(test_input, test_output)
    if net.verbose:
        print 'Cross-validation {0}: train-error {1:.2e}, test-error {2:.2e}\n'.format(iteration,train_error,test_error),
    return net

def train(net , train_input, train_output, iteration, in_cv, test_input = None, test_output = None):
    net.randomweights()
    net.train_cg(train_input, train_output, disp = 0)
    weights = net.weights
    train_error = net.sqerror(train_input,train_output)
    if in_cv:
        test_error = net.sqerror(test_input,test_output)
    else:
        test_error = None
    if net.verbose == 2:
        if in_cv:
            print 'Optimisation {0}: train-error {1:.2e}, test-error {2:.2e}'.format(iteration,train_error,test_error)
        else:
            print 'Optimisation {0}: train-error {1:.2e}'.format(iteration,train_error)
    return net, train_error, test_error


class NeuralNetwork(object):

    def __init__(self, arch, verbose = False):
        """
        Initialises the neural network
        """
        self.verbose = verbose
        self.arch = arch
        conec = mlgraph(self.arch, biases=True)
        self._net = ffnet(conec)

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
            nsample = np.shape(input)[0]
            weights = []
            train_error = []
            test_error = []
            net = deepcopy(self)
            self.cv_nets = Parallel(n_jobs = nproc, verbose = 0)(
                delayed(cross_validation)(net, input, output, ratio = ratio, ntry = ntry, iteration = i, nproc = nproc)
                    for i in range(1,ncv+1))

    def random_split(self, input, output, ratio = 0.75):
        """
        Splits arrays in two datasets train and test according to the ratio specified
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
        Finds best neural network amongst ntry trained neural networks
        The training method is a conjugate gradient descent
        The fitness is measured by sqerror
        """

        # Because the parallel loop copies the object self, we have to add this extra line.
        train_input, train_output = self._setnorm(input = train_input, target = train_output)

        results = Parallel(n_jobs = nproc, verbose = 0)(
                delayed(train)(self, train_input, train_output, iteration = i, in_cv = in_cv, test_input = test_input, test_output = test_output)
                    for i in range(1,ntry+1))

        nets = [result[0] for result in results]
        train_error = [result[1] for result in results]
        test_error = [result[2] for result in results]

        if self.verbose == 2:
            if in_cv:
                print 'Optimisation finished: train-error {0:.2e}, test-error {1:.2e}'.format(np.min(train_error),np.min(test_error))
            else:
                print 'Optimisation finished: train-error {0:.2e}'.format(np.min(train_error))
        self.best_net = nets[np.argmin(train_error)]


    def sim(self, input, cv = False, best = False, net = None):
        """
        Simulates the result of the neural network obtained by either training method
        """
        if cv:
            output_list = []
            for i in range(self.ncv):
                output_list += [self.cv_nets[i].call(input)]
            output = np.concatenate(output_list, axis = 1)
            mean_output = np.mean(output,1)
            nsample = np.shape(input)[0]
            return np.resize(mean_output,(nsample,self.arch[-1]))
        if best:
            return self.best_net.call(input)
        if net is not None:
            return net.call(input)
        return self.call(input)
