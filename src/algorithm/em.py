'''
The implementation of Bayesian Model for streaming data truth detection presented in
Xin Luna Dong, Laure Berti-Equille, Divesh Srivastava
Data Fusion: Resolvnig Conflicts from Multiple Sources

@author: Evgeny Krivosheev (e.krivoshe@gmail.com)
'''

import numpy as np
import pandas as pd
import copy
import random
from common import get_metrics, get_accuracy_err

max_rounds = 100
eps = 10e-3


def get_accuracy(data, prob, sources):
    accuracy_list = []
    for s in sources:
        p_sum = 0.
        size = 0.
        for obj_index in data.keys():
            obj_data = data[obj_index]
            if s not in obj_data[0]:
                continue
            obj_possible_values = sorted(set(obj_data[1]))
            observed_val = obj_data[1][obj_data[0].index(s)]
            val_ind = obj_possible_values.index(observed_val)
            p_sum += prob[obj_index][val_ind]
            size += 1
        accuracy = p_sum/size
        accuracy_list.append(accuracy)

    return accuracy_list


def get_factor(accuracy, v, v_true, n):
    if v == v_true:
        factor = accuracy
    else:
        factor = (1 - accuracy)/n
    return factor


def get_prob(data, accuracy_list, sources):
    likelihood = {}
    for obj_index in data.keys():
        obj_data = data[obj_index]
        possible_values = sorted(set(obj_data[1]))
        l = len(possible_values)
        n = l - 1
        prob = []
        term_list = [1]*l
        for s_ind, v in zip(obj_data[0], obj_data[1]):
            accuracy = accuracy_list[s_ind]
            for v_ind, v_true in enumerate(possible_values):
                term_list[v_ind] *= get_factor(accuracy, v, v_true, n)
        denom = sum(term_list)
        for v_ind in range(l):
            prob.append(term_list[v_ind]/denom)
        likelihood.update({obj_index: prob})
    return likelihood


def em(data=None, gt=None, accuracy_truth=None, s_number=None):
    accuracy_all = []
    sources = range(s_number)
    accuracy_list = [random.uniform(0.8, 0.95) for i in range(s_number)]
    accuracy_delta = 0.3
    iter_number = 0
    while accuracy_delta > eps and iter_number < max_rounds:
        prob = get_prob(data=data, accuracy_list=accuracy_list, sources=sources)
        accuracy_prev = copy.copy(accuracy_list)
        accuracy_list = get_accuracy(data=data, prob=prob, sources=sources)
        accuracy_delta = max([abs(k-l) for k, l in zip(accuracy_prev, accuracy_list)])
        iter_number += 1

    accuracy_all.append(accuracy_list)
    dist_metric, precision = get_metrics(data=data, gt=gt, prob=prob)

    accuracy_mean = []
    accuracy_df = pd.DataFrame(data=accuracy_all)
    for s in range(len(accuracy_list)):
        accuracy_mean.append(np.mean(accuracy_df[s]))
    if accuracy_truth is not None:
        accuracy_err = get_accuracy_err(acc_truth=accuracy_truth, acc=accuracy_list)
    else:
        accuracy_err = None

    return [dist_metric, iter_number, precision, accuracy_err]
