# Copyright (c) Timur Iskhakov.


import pandas
import numpy
import math

from sklearn.feature_extraction import text as sktext
from sklearn import cross_validation
from sklearn import datasets
from sklearn import grid_search
from sklearn import pipeline
from sklearn import svm


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


newsgroups = datasets.fetch_20newsgroups(
    subset='all', 
    categories=['alt.atheism', 'sci.space']
)

pipe = pipeline.Pipeline([
    ('tfidf', sktext.TfidfVectorizer()),
    ('svс', svm.SVC(kernel='linear', random_state=241)),
])

parameters = {'svс__C': [math.pow(10, i) for i in range(-5, 6)]}
kfold = cross_validation.KFold(len(newsgroups.data), n_folds=5, shuffle=True, random_state=241)

clf = grid_search.GridSearchCV(pipe, parameters, scoring='roc_auc', cv=kfold, n_jobs=8)
clf.fit(newsgroups.data, newsgroups.target)

print(clf.best_params_)

estimator = clf.best_estimator_
estimator.fit(newsgroups.data, newsgroups.target)

words = estimator.named_steps['tfidf'].get_feature_names()
array = numpy.absolute(estimator.named_steps['svс'].coef_.toarray())[0]

indices = numpy.argsort(array)[::-1][:10]
important_words = list(map(lambda idx: words[idx], indices))

print_ans(*sorted(important_words), sep=',')
