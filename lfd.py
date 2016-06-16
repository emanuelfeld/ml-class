#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random as rd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def error_rate(hypothesis, actual):
    '''
    return the binary error rate, having
    evaluated a hypothesis against the true values
    '''

    errors = np.not_equal(hypothesis, actual)
    e_rate = np.mean(errors, axis=0)
    return e_rate


def classify(X, weights):
    return np.sign(np.dot(X, weights))


def plot2D(X, Y, weights):
    x = np.linspace(-1,1,100)
    colors = ['red' if Y[i] == 1 else 'blue' for i in range(Y.size)]
    plt.scatter(X[:, 1], X[:, 2], c=colors)
    plt.plot(-1.0 * (self.weights[0] + x * self.weights[1]) / self.weights[2])
    plt.show()


class Line:
    def __init__(self):
        p1 = [rd.uniform(-1, 1), rd.uniform(-1, 1)]
        p2 = [rd.uniform(-1, 1), rd.uniform(-1, 1)]
        self.weights = [
            p1[0] * (p2[1] - p1[1]) + p1[1] * (p1[0] - p2[0]),
            p1[1] - p2[1],
            p2[0] - p1[0]
        ]


class Data:
    def __init__(self, dim=[10, 3], intercept=True):
        assert len(dim) == 2
        assert all(type(d) == int for d in dim)
        
        self.N = dim[0]
        self.width = dim[1]
        self.X = np.empty([self.N, self.width])
        self.Y = np.empty([self.N])
        self.intercept = intercept

        self.generate_data()

    def generate_data(self):
        '''
        create N points with xi0 = 1 and xi1, xi2 ∈ (-1, 1)
        for i ∈ [0, N)
        '''

        for i in range(self.N):
            j = 0
            if self.intercept:
                self.X[i][0] = 1
                j = 1
            while j < self.width:
                self.X[i][j] = rd.uniform(-1, 1)
                j += 1
                
    def add_columns(self, cols):
        '''
        modify dataset to include additional columns
        '''

        for col in cols:
            assert col.size == self.N
            assert col.shape[0] == self.N
            self.X = np.append(self.X, col.reshape([self.N,1]), 1)

    def add_noise(self, fraction=0.1):
        '''
        flip the sign of the output for a given fraction of points
        '''

        for i in rd.sample(range(self.Y.size), int(self.Y.size * fraction)):
            self.Y[i] = -1 * self.Y[i]


class PLA:
    def __init__(self, weights=np.zeros(3), alpha=1.0):
        self.weights = weights
        self.alpha = alpha

    def update_weights(self, x, y):
        self.weights = np.add(self.weights, self.alpha * y * x)

    def run(self, X, Y):
        '''
        update weights until no points are misclassified
        '''

        iterations = 0
        prediction = classify(X, self.weights)
        misclassified = [i for i in range(X.shape[0]) if Y[i] != prediction[i]]

        while misclassified:
            iterations += 1
            # choose a random misclassified point
            misclassified_index = rd.choice(misclassified)
            self.update_weights(X[misclassified_index], Y[misclassified_index])

            prediction = classify(X, self.weights)
            misclassified = [i for i in range(X.shape[0]) if Y[i] != prediction[i]]

        return iterations


class Pocket:
    '''
    pocket algorithm with ratchet, a modification of the perceptron
    learning algorithm, via:
    http://ftp.cs.nyu.edu/~roweis/csc2515-2006/readings/gallant.pdf

    includes a strict iteration limit, whereas Gallant proposes one
    that may be increased
    '''

    def __init__(self, weights=np.zeros(3), iteration_limit=100):
        self.max_iterations = iteration_limit
        self.iterations = 0

        self.pk_weights = weights
        self.pk_streak = 0
        self.pk_correct = 0

        self.pc_weights = weights
        self.pc_streak = 0
        self.pc_correct = 0

    def update_weights(self, x, y):
        self.pc_weights = np.add(self.pc_weights, y * x)

    def run(self, X, Y):
        '''
        update weights until either no points are misclassified
        or the iteration limit has been reached
        '''

        prediction = classify(X, self.pc_weights)
        misclassified = [i for i in range(X.shape[0]) if Y[i] != prediction[i]]

        while self.iterations < self.max_iterations:

            # end if determine training points separable, i.e. Ein is 0
            if not misclassified:  
                break

            self.iterations += 1
            point = rd.choice(range(X.shape[0]))

            if point not in misclassified:
                self.pc_streak += 1
                if self.pc_streak > self.pk_streak:
                    self.pc_correct = X.shape[0] - len(misclassified)
                    if self.pc_correct > self.pk_correct:
                        self.pk_weights = self.pc_weights
                        self.pk_streak = self.pc_streak
                        self.pk_correct = self.pc_correct
            else:
                self.update_weights(X[point], Y[point])
                self.pc_streak = 0

            prediction = classify(X, self.pc_weights)
            misclassified = [i for i in range(X.shape[0]) if Y[i] != prediction[i]]

        return self.pk_weights, self.iterations


class PocketOnline:
    '''
    online (∞) pocket algorithm with ratchet
    http://ftp.cs.nyu.edu/~roweis/csc2515-2006/readings/gallant.pdf
    '''

    def __init__(self, weights=np.zeros(3)):
        self.iterations = 0

        self.pk_weights = weights
        self.pk_streak = 0

        self.pc_weights = weights
        self.pc_streak = 0

    def update_weights(self, x, y):
        self.pc_weights = np.add(self.pc_weights, y * x)

    def run(self, x, y):
        prediction = classify(x, self.pc_weights)

        if prediction == y:
            self.pc_streak += 1
            if self.pc_streak > self.pk_streak:
                self.pk_weights = self.pc_weights
                self.pk_streak = self.pc_streak
        else:
            self.update_weights(x, y)
            self.pc_streak = 0

        self.iterations += 1

        return self.pk_weights, self.iterations


class Winnow:
    def __init__(self):
        pass


class OLS:
    '''
    ordinary least squares regression
    '''

    def __init__(self):
        pass

    def run(self, X, Y):
        self.weights = np.dot(np.linalg.pinv(X), Y)
