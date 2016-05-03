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

max_rounds = 300
eps = 10e-5


def init_prob(data, values):
    init_prob = []
    l = len(values)
    obj_index_list = sorted(data.O.drop_duplicates())
    for obj_index in obj_index_list:
        init_prob.append([1./l]*l)
    return init_prob


def get_accuracy(data, prob, s_number):
    accuracy_list = []
    for s_index in range(s_number):
        p_sum = 0.
        size = 0.
        for psi in data[data.S == s_index].iterrows():
            psi = psi[1]
            observed_val = psi.V
            p_sum += prob[psi.O][observed_val]
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


def get_prob(data, truth_obj_list, accuracy_list, values):
    likelihood = []
    l = len(values)
    n = l - 1
    for obj_index in range(len(truth_obj_list)):
        prob = []
        term_list = [1]*l
        for inst in data[data.O == obj_index].iterrows():
            accuracy = accuracy_list[inst[1].S]
            v = inst[1].V
            for v_ind, v_true in enumerate(values):
                term_list[v_ind] *= get_factor(accuracy, v, v_true, n)
        denom = sum(term_list)
        for v_ind in range(l):
            prob.append(term_list[v_ind]/denom)
        likelihood.append(prob)
    return likelihood


def get_gt_prob(data, truth_obj_list, values):
    prob_gt = []
    val = []
    l = len(values)
    for obj_index in range(len(data.O.drop_duplicates())):
        val.append(values)
        prob_gt.append([0]*l)
    for obj_ind, v_true in enumerate(truth_obj_list):
        for v_ind, v in enumerate(val[obj_ind]):
            if v == v_true:
                prob_gt[obj_ind][v_ind] = 1
    return prob_gt


def get_dist_metric(prob_gt, prob):
    prob_gt_vector = []
    prob_vector = []
    for i in range(len(prob_gt)):
        prob_gt_vector += prob_gt[i]
        prob_vector += prob[i]
    dist_metric = np.dot(prob_gt_vector, prob_vector)
    dist_metric_norm = dist_metric/len(prob_gt)
    return dist_metric_norm


def em(data, truth_obj_list, values):
    dist_list = []
    iter_list = []
    accuracy_all = []
    for round in range(1):
        prob = init_prob(data=data, values=values)
        s_number = len(data.S.drop_duplicates())
        accuracy_list = [random.uniform(0.6, 0.95) for i in range(s_number)]
        accuracy_delta = 0.3
        iter_number = 0
        while accuracy_delta > eps and iter_number < max_rounds:
            prob = get_prob(data=data, truth_obj_list=truth_obj_list, accuracy_list=accuracy_list, values=values)
            accuracy_prev = copy.copy(accuracy_list)
            accuracy_list = get_accuracy(data=data, prob=prob, s_number=s_number)
            accuracy_delta = max([abs(k-l) for k, l in zip(accuracy_prev, accuracy_list)])
            iter_number += 1

        prob_gt = get_gt_prob(data=data, truth_obj_list=truth_obj_list, values=values)
        dist_metric = get_dist_metric(prob_gt=prob_gt, prob=prob)
        dist_list.append(dist_metric)
        iter_list.append(iter_number)
        accuracy_all.append(accuracy_list)

    accuracy_mean = []
    accuracy_df = pd.DataFrame(data=accuracy_all)
    for s in range(len(accuracy_list)):
        accuracy_mean.append(np.mean(accuracy_df[s]))

    return [np.mean(dist_list), np.mean(iter_list), accuracy_mean]
