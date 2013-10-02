import gzip
import cPickle

import numpy as np
import matplotlib.pyplot as plt
from math import exp, log

from time import time

def load_mnist():
    f = gzip.open('mnist.pkl.gz', 'rb')
    data = cPickle.load(f)
    f.close()
    return data

def plot_digits(data, numcols, shape=(28,28)):
    numdigits = data.shape[0]
    numrows = int(numdigits/numcols)
    for i in range(numdigits):
        plt.subplot(numrows, numcols, i)
        plt.axis('off')
        plt.imshow(data[i].reshape(shape), interpolation='nearest', cmap='Greys')
    plt.show()

def plot_digits2(data, numcols, shape=(28,28)):
    data = data.T
    numdigits = len(data)
    numrows = int(numdigits/numcols)
    for i in range(numdigits):
        plt.subplot(numrows, numcols, i)
        plt.axis('off')
        plt.imshow(data[i].reshape(shape), interpolation='nearest', cmap='Greys')
    plt.show()
    
"assignemnt 1.112 where the fucntions of 1.111 are filled in. %TODO: if something changes in 1.111 we have to change this formula!" 
def logreg_gradient(x, t, w, b):
    matrixW = np.zeros(w.shape).T #transposed to fill the matrix easier (per row instead of column).
    vectorB = np.zeros(b.shape).T

    wT = w.T
    bT = b.T

    Z = sum([exp(wT[k].dot(x) + b[k]) for k in xrange(len(matrixW))])

    for column in xrange(len(matrixW)):
        q = exp(wT[column].dot(x) + bT[column])

        if column == t:
            gradW = x - ( (q*x)/Z )
            gradB = 1 - ( q/Z )
            logP = log(q) - log(Z)
        else:
            gradW = -( (q*x)/Z )
            gradB = -( q/Z )

        vectorB[column] = gradB
        matrixW[column] = gradW

    return matrixW.T, vectorB.T #transpose them back to give the right formula back.


"Assignment 1.1.3, maar mijn aannames moeten nog worden gecheckt."
def sgd_iter(x_train, t_train, w, b):
    learn = 0.0001

    all_indices = range(len(x_train))
    np.random.shuffle(all_indices)

    for i in all_indices:
        gradW, gradB = logreg_gradient(x_train[i], t_train[i], w, b)
        w += learn*gradW
        b += learn*gradB
    return w, b

def logPData(x, t, w, b):
    Z = sum( [exp(w.T[k].dot(x) + b[k]) for k in xrange(len(w.T))] )
    q = exp(w.T[t].dot(x) + b.T[t])
    return log(q)-log(Z)


def main(x_train, t_train, x_valid, t_valid, w, b, training_iterations=4):
    start = time()

    logP = np.zeros(training_iterations)
    logPvalid = np.zeros(training_iterations)

    # This will be filled on the last training iteration to display the low8 and high8
    last_iteration_values = []

    for i in xrange(training_iterations):
        w, b = sgd_iter(x_train, t_train, w, b)

        for j in xrange(len(x_train)):
            logP[i] += logPData(x_train[j], t_train[j], w, b)

        for j in xrange(len(x_valid)):
            v = logPData(x_valid[j], t_valid[j], w, b)
            logPvalid[i] += v

            if i == training_iterations-1:
                last_iteration_values.append((x_valid[j], t_valid[j], v))

    stop = time()

    print "Time: ", int(stop-start+0.5)

    logP = logP / len(x_train)
    logPvalid = logPvalid / len(x_valid)

    last_iteration_values.sort(key = lambda tup : tup[2])

    plot_digits(np.array(map(lambda x : x[0], last_iteration_values[0:8])), numcols = 4)
    plot_digits(np.array(map(lambda x : x[0], last_iteration_values[len(last_iteration_values)-8:])), numcols = 4)

    plt.plot(range(1,training_iterations+1), logP, range(1,training_iterations+1), logPvalid)
    plt.plot(range(1,training_iterations+1), logP, 'o', range(1,training_iterations+1), logPvalid, 'bs')
    plt.show()

    #vraag 1.2.1
    plot_digits2(w, numcols=5)
    plt.show()

if __name__ == "__main__":
    """ Load program data and start """
    (x_train, t_train), (x_valid, t_valid), (x_test, t_test) = load_mnist()

    #Prepare our weights
    w = np.zeros((784, 10))
    b = np.zeros(10).T

    training_iterations = 4
    small = False
    if small == False:
        main(x_train, t_train, x_valid, t_valid, w, b, training_iterations)
    else:
        main(x_train[0:small], t_train[0:small], x_valid[0:small], t_valid[0:small], w, b, training_iterations)

    
