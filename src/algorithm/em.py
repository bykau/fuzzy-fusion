'''
The implementation of Bayesian Model for streaming data truth detection presented in
Xin Luna Dong, Laure Berti-Equille, Divesh Srivastava
Data Fusion: Resolvnig Conflicts from Multiple Sources

@author: Evgeny Krivosheev (e.krivoshe@gmail.com)
'''

import numpy as np
import copy

max_rounds = 30
eps = 0.001


def get_accuracy(data, prob, s_number):
    accuracy_list = []
    for s_index in range(s_number):
        p_sum = 0.
        size = 0.
        for obj_index in range(len(data.O.drop_duplicates())):
            observed_val = list(data[(data.S == s_index) & (data.O == obj_index)].V)
            if len(observed_val) != 0:
                observed_val = observed_val[0]
            else:
                continue
            p_sum += prob[obj_index][observed_val]
            size += 1
        accuracy = p_sum/size
        accuracy_list.append(accuracy)
    return accuracy_list


def get_prob(data, accuracy, truth_obj_list, accuracy_list):
    likelihood = []
    for obj_index in range(len(truth_obj_list)):
        a, b = 1., 1.
        v_possible = 0
        for inst in data[data.O == obj_index].iterrows():
            accuracy = accuracy_list[inst[1].S]
            # TO DO !!!
            if accuracy == 1:
                accuracy = 0.95
            v = inst[1].V
            if v == v_possible:
                b *= accuracy/(1-accuracy)
            else:
                a *= accuracy/(1-accuracy)
        likelihood.append([b/(a+b), a/(a+b)])
    return likelihood


def get_gt_prob(data, truth_obj_list):
    prob_gt = []
    val = []
    for obj_index in range(len(data.O.drop_duplicates())):
        possible_values = [0, 1]
        val.append(possible_values)
        prob_gt.append([0]*len(possible_values))
    for obj_ind, v_true in enumerate(truth_obj_list):
        for v_ind, v in enumerate(val[obj_ind]):
            if v == v_true:
                prob_gt[obj_ind][v_ind] = 1
    return prob_gt, val


def get_dist_metric(prob_gt, prob):
    prob_gt_vector = []
    prob_vector = []
    for i in range(len(prob_gt)):
        prob_gt_vector += prob_gt[i]
        prob_vector += prob[i]
    dist_metric = np.dot(prob_gt_vector, prob_vector)
    dist_metric_norm = dist_metric/len(prob_gt)
    return dist_metric_norm


def em(data, accuracy, truth_obj_list):
    s_number = len(accuracy.S)
    accuracy_list = list(accuracy.A)
    accuracy_delta = 0.3
    iter_number = 0
    while accuracy_delta > eps and iter_number < max_rounds:
        prob = get_prob(data=data, accuracy=accuracy_list, truth_obj_list=truth_obj_list, accuracy_list=accuracy_list)
        accuracy_prev = copy.copy(accuracy_list)
        accuracy_list = get_accuracy(data=data, prob=prob, s_number=s_number)
        accuracy_delta = max([abs(k-l) for k, l in zip(accuracy_prev, accuracy_list)])
        iter_number += 1

    prob_gt, val = get_gt_prob(data=data, truth_obj_list=truth_obj_list)
    dist_metric = get_dist_metric(prob_gt=prob_gt, prob=prob)

    return [dist_metric, iter_number]
