"""
Contains the functions to compute performance metrics of algorithms.
"""
from __future__ import division
import operator


def error_rate(pred, actu):
    """
    The percentage of false values.
    
    :param pred: predicted probs
    :param actu: ground truth
    :return: error rate
    """
    n_errors = 0
    for obj in actu.keys():
        dominant = max(pred[obj].iteritems(), key=operator.itemgetter(1))[0]
        if dominant != actu[obj]:
            n_errors += 1

    return n_errors/len(actu.keys())
