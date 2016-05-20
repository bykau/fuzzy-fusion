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
from common import get_dist_metric, get_precision, get_accuracy_err

max_rounds = 100
eps = 10e-3


def get_accuracy(data, prob, sources):
    accuracy_list = []
    for s in sources:
        p_sum = 0.
        size = 0.
        for psi in data[data.S == s].iterrows():
            psi = psi[1]
            obj_values = sorted(data[data.O == psi.O].V.drop_duplicates().values)
            observed_val = psi.V
            v_ind = obj_values.index(observed_val)
            p_sum += prob[psi.O][v_ind]
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
    likelihood = []
    for obj_index in range(len(data.O.drop_duplicates())):
        values = sorted(data[data.O == obj_index].V.drop_duplicates().values)
        l = len(values)
        n = l - 1
        prob = []
        term_list = [1]*l
        for inst in data[data.O == obj_index].iterrows():
            s_ind = sources.index(inst[1].S)
            accuracy = accuracy_list[s_ind]
            v = inst[1].V
            for v_ind, v_true in enumerate(values):
                term_list[v_ind] *= get_factor(accuracy, v, v_true, n)
        denom = sum(term_list)
        for v_ind in range(l):
            prob.append(term_list[v_ind]/denom)
        likelihood.append(prob)
    return likelihood


def em(data, truth_obj_list, accuracy_truth):
    accuracy_all = []
    sources = sorted(data.S.drop_duplicates().values)
    s_number = len(sources)
    accuracy_list = [random.uniform(0.8, 0.95) for i in range(s_number)]
    accuracy_delta = 0.3
    iter_number = 0
    while accuracy_delta > eps and iter_number < max_rounds:
        prob = get_prob(data=data, accuracy_list=accuracy_list , sources=sources)
        accuracy_prev = copy.copy(accuracy_list)
        accuracy_list = get_accuracy(data=data, prob=prob, sources=sources)
        accuracy_delta = max([abs(k-l) for k, l in zip(accuracy_prev, accuracy_list)])
        iter_number += 1

    dist_metric = get_dist_metric(data=data, truth_obj_list=truth_obj_list, prob=prob[0:len(truth_obj_list)])
    accuracy_all.append(accuracy_list)
    precision = get_precision(data=data, truth_obj_list=truth_obj_list, prob=prob[0:len(truth_obj_list)])

    accuracy_mean = []
    accuracy_df = pd.DataFrame(data=accuracy_all)
    for s in range(len(accuracy_list)):
        accuracy_mean.append(np.mean(accuracy_df[s]))
    accuracy_err = get_accuracy_err(acc_truth=accuracy_truth, acc=accuracy_list)

    return [dist_metric, iter_number, precision, accuracy_err]
