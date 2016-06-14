"""
Gibbs sampling truth finder

@author: Evgeny Krivosheev (e.krivoshe@gmail.com)
"""


import numpy as np
import pandas as pd
from scipy.stats import beta
import copy
import random
from common import get_metrics, get_accuracy_err

max_rounds = 5
alpha1, alpha2 = 1, 1


def init_var(data, s_number):
    s_index_list = range(s_number)
    obj_index_list = data.keys()
    var_index = [obj_index_list, s_index_list]
    accuracy_list = [random.uniform(0.8, 0.95) for i in range(s_number)]
    obj_values = {}
    init_prob = {}
    counts = {}
    for obj_index in obj_index_list:
        values = data[obj_index][1]
        possible_values = sorted(set(values))
        obj_val = max(set(values), key=values.count)
        obj_values.update({obj_index: obj_val})
        l = len(possible_values)
        init_prob.update({obj_index: [1./l]*l})

        sources = data[obj_index][0]
        counts_list = []
        for val in values:
            if val == obj_val:
                counts_list.append(1)
            else:
                counts_list.append(0)
        counts.update({obj_index: [sources, counts_list]})

    return [var_index, obj_values, counts, init_prob, accuracy_list]


def get_o(o_ind, obj_values, counts, obj_data, accuracy_list):
    l_p = []
    l_c = []
    sources = obj_data[0]
    values = obj_data[1]
    possible_values = sorted(set(values))
    n = len(possible_values) - 1
    for v in possible_values:
        pr = 1
        counts_v = copy.deepcopy(counts)
        for psi_ind, psi in enumerate(values):
            s = sources[psi_ind]
            s_accuracy = accuracy_list[s]
            if psi == v:
                pr *= s_accuracy
                c_new = 1
            else:
                pr *= (1 - s_accuracy)/n
                c_new = 0

            c_old = counts[o_ind][1][psi_ind]
            if c_new != c_old:
                counts_v[o_ind][1][psi_ind] = c_new
        l_p.append(pr)
        l_c.append(counts_v)
    norm_const = sum(l_p)
    for v_ind in range(len(possible_values)):
        l_p[v_ind] /= norm_const
    mult_trial = list(np.random.multinomial(1, l_p, size=1)[0])
    v_new_ind = mult_trial.index(1)
    v_new = possible_values[v_new_ind]
    obj_values.update({o_ind: v_new})
    counts_new = l_c[v_new_ind]

    return [counts_new, l_p]


def get_a(data, s, counts):
    count_p = 0
    count_m = 0
    obj_index_list = data.keys()
    for obj_index in obj_index_list:
        obj_data = data[obj_index]
        sources = obj_data[0]
        if s not in sources:
            continue
        obj_counts = counts[obj_index][1]
        s_index = sources.index(s)
        c = obj_counts[s_index]
        if c == 1:
            count_p += 1
        else:
            count_m += 1
    a_new = beta.rvs(count_p + alpha1, count_m + alpha2, size=1)[0]

    return a_new


def gibbs(data=None, gt=None, accuracy_truth=None, s_number=None):
    accuracy_all = []
    var_index, obj_values, counts, prob, accuracy_list = init_var(data=data, s_number=s_number)
    iter_number = 0
    dist_temp = []
    precision_temp = []
    while iter_number < max_rounds:
        for o_ind in var_index[0]:
            obj_data = data[o_ind]
            # obj_values[o_ind], counts, prob[o_ind] = \
            counts, prob_new = get_o(o_ind=o_ind, obj_values=obj_values, counts=counts,
                                     obj_data=obj_data, accuracy_list=accuracy_list)
            prob.update({o_ind: prob_new})
        for s in var_index[1]:
            accuracy_list[s] = get_a(data=data, s=s, counts=counts)

        iter_number += 1
        dist_metric, precision = get_metrics(data=data, gt=gt, prob=prob)
        dist_temp.append(dist_metric)
        precision_temp.append(precision)
        accuracy_all.append(accuracy_list)

    dist_metric = np.mean(dist_temp[-3:])
    precision = np.mean(precision_temp[-3:])

    accuracy_mean = []
    accuracy_df = pd.DataFrame(data=accuracy_all)
    for s in range(len(accuracy_list)):
        accuracy_mean.append(np.mean(accuracy_df[s]))
    if accuracy_truth is not None:
        accuracy_err = get_accuracy_err(acc_truth=accuracy_truth, acc=accuracy_list)
    else:
        accuracy_err = None

    return [dist_metric, iter_number, precision, accuracy_err]
