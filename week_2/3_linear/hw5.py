# Copyright (c) Timur Iskhakov.


import pandas
import numpy
import math

import sklearn.preprocessing
import sklearn.linear_model


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


train = read_dataset('perceptron-train.csv')
test = read_dataset('perceptron-test.csv')

clf = sklearn.linear_model.Perceptron(random_state=241)
clf.fit(train[0], train[1])
first_score = clf.score(test[0], test[1])

scaler = sklearn.preprocessing.StandardScaler()
train[0] = scaler.fit_transform(train[0])
test[0] = scaler.transform(test[0])

clf.fit(train[0], train[1])
second_score = clf.score(test[0], test[1])

print_ans(round(second_score - first_score, 3))
