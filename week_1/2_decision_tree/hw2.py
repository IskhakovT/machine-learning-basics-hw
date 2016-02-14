# Copyright (c) Timur Iskhakov.


import pandas
import numpy
import math

from sklearn.tree import DecisionTreeClassifier


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


data = pandas.read_csv('titanic.csv', index_col='PassengerId')

matrix = []
survived = []

features = ('Pclass', 'Fare', 'Age', 'Sex', 'Survived')

def proc_row(row):
    ret = []
    for feature in features:
        if type(row[feature]) == float and math.isnan(row[feature]):
            return None
        if feature == 'Sex':
            ret.append(1. if row[feature] == 'male' else 2.)
        else:
            ret.append(float(row[feature]))
    return ret

for index, row in data.iterrows():
    curr = proc_row(row)
    if curr:
        matrix.append(curr[:-1])
        survived.append(curr[-1])

clf = DecisionTreeClassifier(random_state=241)
clf.fit(numpy.array(matrix), numpy.array(survived))

importances = clf.feature_importances_
indices = numpy.argsort(importances)

print_ans(features[indices[-1]], features[indices[-2]], sep=',')
