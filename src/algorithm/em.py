'''
The implementation of Bayesian Model for streaming data truth detection presented in
Xin Luna Dong, Laure Berti-Equille, Divesh Srivastava
Data Fusion: Resolvnig Conflicts from Multiple Sources

@author: Evgeny Krivosheev (e.krivoshe@gmail.com)
'''

import numpy as np
import pandas as pd
import copy

max_rounds = 30
eps = 0.001


def get_n_params(data):
    n_list = []
    obj_list = sorted(data.O.drop_duplicates())
    for i in obj_list:
        n = len(data[data.O == i].V.drop_duplicates()) - 1
        n_list.append(n)
    return n_list


def get_accuracy(data, prob):
    accuracy_list = []
    values = []
    for s_index in range(s_number):
        p_sum = 0.
        size = 0.
        for obj_index in range(len(data.O.drop_duplicates())):
            observed_val = list(data[(data.S == s_index) & (data.O == obj_index)].V)
            if len(observed_val) != 0:
                observed_val == observed_val[0]
            else:
                continue
            size += 1
            possible_values = sorted(list(set(data[data.O == obj_index].V)))
            values.append(possible_values)
            for v_ind, v in enumerate(possible_values):
                if v == observed_val:
                    p_sum += prob[obj_index][v_ind]
                    break
        accuracy = p_sum/size
        accuracy_list.append(accuracy)
    return accuracy_list


def get_prob(data, accuracy):
    n_list = get_n_params(data=data)
    likelihood = []
    for obj_index in range(len(truth_obj_list)):
            likelihood.append([])
            n = n_list[obj_index]
            observed_values = list(data[data.O == obj_index].V)
            possible_values = sorted(list(set(observed_values)))
            if n == 0:
                likelihood[obj_index].append(1.)
                continue
            for v_true in possible_values:
                a, b, b_sum = 1., 1., 0.
                a_not_completed = True
                for v_possible in possible_values:
                    for inst in data[data.O == obj_index].iterrows():
                        accuracy = accuracy_list[inst[1].S]
                        v = inst[1].V
                        if v == v_possible:
                            b *= n*accuracy/(1-accuracy)
                        if a_not_completed and v == v_true:
                            a *= n*accuracy/(1-accuracy)
                    a_not_completed = False
                    b_sum += b
                    b = 1
                p = a/b_sum
                likelihood[obj_index].append(p)
    return likelihood


def get_gt_prob(data, truth_obj_list):
    prob_gt = []
    val = []
    for obj_index in range(len(data.O.drop_duplicates())):
        possible_values = sorted(list(set(data[data.O == obj_index].V)))
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


if __name__ == '__main__':
    data = pd.read_csv('../../data/observation.csv', names=['S', 'O', 'V'])
    accuracy = pd.read_csv('../../data/accuracy.csv', names=['S', 'A'])
    s_number = len(accuracy.S)
    accuracy_list = list(accuracy.A)
    truth_obj_list = [6, 8, 9, 15, 16, 10, 11, 7, 18, 20]

    accuracy_delta = 0.3
    iter_number = 0
    while accuracy_delta > eps and iter_number < max_rounds:
        prob = get_prob(data=data, accuracy=accuracy_list)
        accuracy_prev = copy.copy(accuracy_list)
        accuracy_list = get_accuracy(data=data, prob=prob)
        accuracy_delta = max([abs(k-l) for k, l in zip(accuracy_prev, accuracy_list)])
        iter_number += 1

    prob_gt, val = get_gt_prob(data=data, truth_obj_list=truth_obj_list)
    dist_metric = get_dist_metric(prob_gt=prob_gt, prob=prob)

    print 'max_dist_metr: {}'.format(dist_metric)
    print 'iter number: {}'.format(iter_number)
    print '------------'
    for v, p in zip(val, prob):
        print v
        print p
        print '_____'
