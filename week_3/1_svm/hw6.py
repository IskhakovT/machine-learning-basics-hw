# Copyright (c) Timur Iskhakov.


import pandas
import numpy
import math

import sklearn.svm


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


data = read_dataset('svm-data.csv')

svc = sklearn.svm.SVC(100000, kernel='linear', random_state=241)
svc.fit(data[0], data[1])

indices = [ind + 1 for ind in sorted(svc.support_)]
print_ans(*indices, sep=',')
