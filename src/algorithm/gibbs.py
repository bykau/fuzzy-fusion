"""
Gibbs sampling truth finder

@author: Evgeny Krivosheev (e.krivoshe@gmail.com)
"""


import numpy as np
import copy
import random

max_rounds = 30
eps = 0.001
l = 2


def init_var(data):
    observ_val = []
    accuracy_ind = sorted(data.S.drop_duplicates())
    s_number = len(accuracy_ind)
    accuracy_list = [random.uniform(0.6, 0.95) for i in range(s_number)]
    obj_index_list = sorted(data.O.drop_duplicates())
    for obj_index in obj_index_list:
        possible_values = sorted(list(set(data[data.O == obj_index].V)))
        observ_val.append(possible_values)
    random.shuffle(obj_index_list)
    random.shuffle(accuracy_ind)
    var_index = [obj_index_list, accuracy_ind]
    return [observ_val, var_index, accuracy_list, s_number]


def get_init_prob(data, values):
    init_prob = []
    l = len(values)
    obj_index_list = sorted(data.O.drop_duplicates())
    for obj_index in obj_index_list:
        init_prob.append([1./l]*l)
    return init_prob


def get_factor(data, accuracy, a_val, v, v_true, n):
    factor = 0.
    if a_val:
        if v == v_true:
            factor = accuracy
    else:
        if v != v_true:
            factor = (1-accuracy)/n
    if factor == 0:
        a_val = 1 - a_val
        factor = get_factor(data, accuracy, a_val, v, v_true, n)
    return factor


def get_prob(data, accuracy_list, obj_index, values):
    prob = []
    l = len(values)
    n = l - 1
    term_list = [1]*l
    for psi in data[data.O == obj_index].iterrows():
        v = psi[1].V
        accuracy = accuracy_list[psi[1].S]
        if accuracy == 0.5:
            a_val = random.choice([0, 1])
        else:
            if accuracy > 0.5:
                a_val = 1
            else:
                a_val = 0
        for v_ind, v_true in enumerate(values):
            term_list[v_ind] *= get_factor(data, accuracy, a_val, v, v_true, n)
    denom = sum(term_list)
    for v_ind in range(l):
        prob.append(term_list[v_ind]/denom)
    return prob


def get_accuracy(data, prob, s_index):
    p_sum = 0.
    size = 0.
    for psi in data[data.S == s_index].iterrows():
        psi = psi[1]
        observed_val = psi.V
        p_sum += prob[psi.O][observed_val]
        size += 1
    accuracy = p_sum/size
    return accuracy


def get_dist_metric(data, truth_obj_list, prob, values):
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
    prob_gt_vector = []
    prob_vector = []
    for i in range(len(prob_gt)):
        prob_gt_vector += prob_gt[i]
        prob_vector += prob[i]
    dist_metric = np.dot(prob_gt_vector, prob_vector)
    dist_metric_norm = dist_metric/len(prob_gt)
    return dist_metric_norm


def run_float(scalar, vector_size):
    random_vector = [random.random() for i in range(vector_size)]
    random_vector_sum = sum(random_vector)
    random_vector = [scalar*i/random_vector_sum for i in random_vector]
    return random_vector


def gibbs_sampl(data, truth_obj_list, values):
    dist_list = []
    iter_number_list = []

    for t in range(5):
        observ_val, var_index, accuracy_list, s_number = init_var(data=data)
        prob = get_init_prob(data=data, values=values)
        accuracy_list = [random.uniform(0.6, 0.95) for i in range(s_number)]
        accuracy_delta = 0.3
        iter_number = 0
        while accuracy_delta > eps and iter_number < max_rounds:
            indexes = copy.deepcopy(var_index)
            accuracy_prev = copy.copy(accuracy_list)
            round_compl = False
            while not round_compl:
                if len(indexes[0])!= 0 and len(indexes[1])!= 0:
                    r = random.randint(0, 1)
                    if r == 1:
                        o_ind = indexes[0].pop()
                        prob[o_ind] = get_prob(data=data, accuracy_list=accuracy_list, obj_index=o_ind, values=values)
                    else:
                        s_index = indexes[1].pop()
                        accuracy_list[s_index] = get_accuracy(data=data, prob=prob, s_index=s_index)
                elif len(indexes[0])==0 and len(indexes[1])!=0:
                        s_index = indexes[1].pop()
                        accuracy_list[s_index] = get_accuracy(data=data, prob=prob, s_index=s_index)
                elif len(indexes[0])!=0 and len(indexes[1])==0:
                    o_ind = indexes[0].pop()
                    prob[o_ind] = get_prob(data=data, accuracy_list=accuracy_list, obj_index=o_ind, values=values)
                else:
                    round_compl = True
            iter_number += 1
            accuracy_delta = max([abs(k-l) for k, l in zip(accuracy_prev, accuracy_list)])
        dist_metric = get_dist_metric(data=data, truth_obj_list=truth_obj_list, prob=prob, values=values)
        dist_list.append(dist_metric)
        iter_number_list.append(iter_number)
    return [np.mean(dist_list), np.mean(iter_number_list)]
