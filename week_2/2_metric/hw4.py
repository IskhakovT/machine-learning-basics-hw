# Copyright (c) Timur Iskhakov.


import pandas
import numpy
import math

import sklearn.cross_validation
import sklearn.datasets
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


dataset = sklearn.datasets.load_boston()
features = sklearn.preprocessing.scale(dataset.data)
target = dataset.target

kfold = sklearn.cross_validation.KFold(len(target), n_folds=5, shuffle=True, random_state=42)


def test_knn(p):
    knn = sklearn.neighbors.KNeighborsRegressor(n_neighbors=5, weights='distance', metric='minkowski', p=p)
    res = sklearn.cross_validation.cross_val_score(knn, features, target, scoring='mean_squared_error', cv=kfold)
    return (res.mean(), p)


res = min([test_knn(i) for i in numpy.linspace(1, 10, num=200)])
print_ans(round(res[1], 1))
