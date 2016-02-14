# Copyright (c) Timur Iskhakov.


import pandas
import numpy

from sklearn import metrics
import matplotlib.pyplot as plt


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


def classification():
    """First part of the homework."""

    def locate(df, values):
        ret = df
        for column in values:
            ret = ret.loc[ret[column] == values[column]]
        return ret

    dataset = pandas.read_csv('classification.csv', header=0)

    TP = locate(dataset, {'true': 1, 'pred': 1})
    FP = locate(dataset, {'true': 0, 'pred': 1})
    FN = locate(dataset, {'true': 1, 'pred': 0})
    TN = locate(dataset, {'true': 0, 'pred': 0})

    print_ans(len(TP), len(FP), len(FN), len(TN))

    print_ans(
        round(metrics.accuracy_score(dataset['true'], dataset['pred']), 2),
        round(metrics.precision_score(dataset['true'], dataset['pred']), 2),
        round(metrics.recall_score(dataset['true'], dataset['pred']), 2),
        round(metrics.f1_score(dataset['true'], dataset['pred']), 2),
    )


def score_metrics():
    """Second part of the homework."""

    dataset = pandas.read_csv('scores.csv', header=0)
    metric_names = list(dataset.columns)[1:]

    roc_auc_scores = [metrics.roc_auc_score(dataset['true'], dataset[metric]) for metric in metric_names]
    print_ans(metric_names[numpy.argsort(roc_auc_scores)[-1]])

    max_precisions = []
    for metric in metric_names:
        precision, recall, thresholds = metrics.precision_recall_curve(dataset['true'], dataset[metric])
        max_precisions.append(precision[(recall >= 0.7).nonzero()].max())

        # Additionally draw a graph
        plt.plot(recall, precision, label=metric)
        plt.xlim(0.7, 1)
        plt.ylim(0.5, 1)

    print_ans(metric_names[numpy.argsort(max_precisions)[-1]])

    plt.legend()
    plt.savefig('fig1.pdf')


classification()
score_metrics()
