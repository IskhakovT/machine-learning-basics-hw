# Copyright (c) Timur Iskhakov.


import pandas
import numpy
import math

import sklearn.cross_validation
import sklearn.neighbors
import sklearn.preprocessing


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


data = pandas.read_csv('wine.data', header=None)

classes = data[0]
features = data.ix[:,1:]

kfold = sklearn.cross_validation.KFold(len(data), n_folds=5, shuffle=True, random_state=42)


def test_knn(n_neighbors):
    knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=n_neighbors)
    res = sklearn.cross_validation.cross_val_score(knn, features, classes, scoring='accuracy', cv=kfold)
    return (res.mean(), n_neighbors)


first_res = max([test_knn(i) for i in range(1, 50)])
print_ans(first_res[1])
print_ans(round(first_res[0], 2))

features = sklearn.preprocessing.scale(features)

second_res = max([test_knn(i) for i in range(1, 50)])
print_ans(second_res[1])
print_ans(round(second_res[0], 2))
