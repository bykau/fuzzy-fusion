"""
Gibbs sampling truth finder

@author: Evgeny Krivosheev (e.krivoshe@gmail.com)
"""


import numpy as np
import pandas as pd
from scipy.stats import beta
import copy
import random

max_rounds = 30
possible_values = [0, 1]
alpha1, alpha2 = 1, 1
l = len(possible_values)


def init_var(data):
    s_ind = sorted(data.S.drop_duplicates())
    s_number = len(s_ind)
    obj_index_list = sorted(data.O.drop_duplicates())
    var_index = [obj_index_list, s_ind]
    accuracy_list = [random.uniform(0.6, 0.95) for i in range(s_number)]
    obj_values = [random.choice(possible_values) for i in range(len(obj_index_list))]

    init_prob = []
    counts_list = []
    for obj_index in obj_index_list:
        init_prob.append([1./l]*l)
    for psi in data.iterrows():
        psi = psi[1]
        if psi.V == obj_values[psi.O]:
            counts_list.append(1)
        else:
            counts_list.append(0)
    counts = pd.DataFrame(counts_list, columns=['c'])

    return [var_index, obj_values, counts, init_prob, accuracy_list]


def get_o(o_ind, obj_values, counts, data, prob, accuracy_list):
    l_p = []
    l_c = []
    for v in possible_values:
        pr = prob[o_ind][v]
        counts_v = copy.deepcopy(counts)
        psi_obj = data[data.O == o_ind]
        for psi in psi_obj.iterrows():
            psi_ind = psi[0]
            psi = psi[1]
            if psi.V == v:
                pr *= accuracy_list[psi.S]
                c_new = 1
            else:
                pr *= 1 - accuracy_list[psi.S]
                c_new = 0

            c_old = counts_v.at[psi_ind, 'c']
            if c_new != c_old:
                counts_v.at[psi_ind, 'c'] = c_new
        l_p.append(pr)
        l_c.append(counts_v)
    norm_const = sum(l_p)
    l_p[0] /= norm_const
    l_p[1] /= norm_const
    v_new = np.random.binomial(1, l_p[1], 1)[0]
    counts_new = l_c[v_new]

    return [v_new, counts_new, l_p]


def get_a(s_psi, counts):
    s_counts = counts.loc[s_psi]
    count_p = len(s_counts[s_counts.c == 1])
    count_m = len(s_counts[s_counts.c == 0])
    a_new = beta.rvs(count_p + alpha1, count_m + alpha2, size=1)[0]

    return a_new


def get_dist_metric(data, truth_obj_list, prob):
    prob_gt = []
    val = []
    for obj_index in range(len(data.O.drop_duplicates())):
        val.append(possible_values)
        prob_gt.append([0]*l)
    for obj_ind, v_true in enumerate(truth_obj_list):
        for v_ind, v in enumerate(val[obj_ind]):
            if v == v_true:
                prob_gt[obj_ind][v_ind] = 1
    prob_gt_vector = []
    prob_vector = []
    for i in range(len(prob_gt)):
        prob_gt_vector += prob_gt[i]
        prob_vector += prob[i]
    dist_metric = np.dot(prob_gt_vector, prob_vector)
    dist_metric_norm = dist_metric/len(prob_gt)

    return dist_metric_norm


def gibbs(data, truth_obj_list):
    dist_list = []
    iter_list = []
    accuracy_all = []
    for round in range(5):
        var_index, obj_values, counts, prob, accuracy_list = init_var(data)
        iter_number = 0
        dist_temp = []
        while iter_number < max_rounds:
            for o_ind in var_index[0]:
                obj_values[o_ind], counts, prob[o_ind] = get_o(o_ind=o_ind, obj_values=obj_values,
                                                               counts=counts, data=data,
                                                               prob=prob, accuracy_list=accuracy_list)

            for s_ind in var_index[1]:
                s_psi = data[data.S == s_ind].index
                accuracy_list[s_ind] = get_a(s_psi=s_psi, counts=counts)

            iter_number += 1
            dist_metric = get_dist_metric(data=data, truth_obj_list=truth_obj_list, prob=prob)
            dist_temp.append(dist_metric)

        accuracy_all.append(accuracy_list)
        dist_metric = np.mean(dist_temp[-10:])
        dist_list.append(dist_metric)
        iter_list.append(iter_number)

    accuracy_mean = []
    accuracy_df = pd.DataFrame(data=accuracy_all)
    for s in range(len(accuracy_list)):
        accuracy_mean.append(np.mean(accuracy_df[s]))

    return [np.mean(dist_list), np.mean(iter_list), accuracy_mean]
