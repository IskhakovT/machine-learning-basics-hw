# Copyright (c) Timur Iskhakov.


import pandas
import numpy

from sklearn import metrics


def static_var(varname, value):
    def decorate(func):
        setattr(func, varname, value)
        return func
    return decorate


@static_var('test_num', 1)
def print_ans(*args, sep=' '):
    with open('{}.txt'.format(print_ans.test_num), 'w') as fout:
        fout.write(sep.join(list(map(str, args))))
    print_ans.test_num += 1


def read_dataset(filename):
    dataset = pandas.read_csv(filename, header=None)
    return [dataset.ix[:,1:], dataset[0]]


def delta(X, y, w, c, k):
    dp = numpy.einsum('ij,j->i', X, w)
    coeff = 1 - (1 / (1 + numpy.exp(-y*dp)))
    a = k / coeff.shape[0] * (numpy.einsum('ij,i,i->j', X, y,coeff))
    return w + a - k*c*w


def regression(X, y, w_init, c, k, EPS=1e-6):
    w_last, w_curr = w_init, delta(X, y, w_init, c, k)
    while numpy.linalg.norm(w_last - w_curr) > EPS:
        w_last, w_curr = w_curr, delta(X, y, w_curr, c, k)
    return w_curr


data = read_dataset('data-logistic.csv')

first_weights = regression(data[0], data[1], numpy.array((0., 0.)), 0, 0.1)
first_res = numpy.dot(data[0], first_weights)
first_ans = metrics.roc_auc_score(data[1], first_res)

second_weights = regression(data[0], data[1], numpy.array((0., 0.)), 10., 0.1)
second_res = numpy.dot(data[0], second_weights)
second_ans = metrics.roc_auc_score(data[1], second_res)

print_ans(round(first_ans, 3), round(second_ans, 3))
