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


def init_var(data, accuracy):
    observ_val = []
    s_number = len(accuracy.S)
    accuracy_list = list(accuracy.A)
    accuracy_ind = sorted(data.S.drop_duplicates())
    obj_index_list = sorted(data.O.drop_duplicates())
    for obj_index in obj_index_list:
        possible_values = sorted(list(set(data[data.O == obj_index].V)))
        observ_val.append(possible_values)
    random.shuffle(obj_index_list)
    random.shuffle(accuracy_ind)
    var_index = [obj_index_list, accuracy_ind]
    return [observ_val, var_index, accuracy_list, s_number]


def get_init_prob(data):
    init_prob = []
    obj_index_list = sorted(data.O.drop_duplicates())
    for obj_index in obj_index_list:
        # init_prob.append(run_float(scalar=1, vector_size=l))
        init_prob.append([0.5, 0.5])
    return init_prob


def get_factor(data, accuracy, v, v_true, p_v_obj):
    p_true = p_v_obj[v_true]
    if v == v_true:
        factor = accuracy*p_true
    else:
        factor = (1 - accuracy)*p_true
    return factor


def get_prob(data, accuracy_list, obj_index, prob_old):
    possible_values = [0, 1]
    a, b = 1., 1.
    p_v_obj = prob_old[obj_index]
    for inst in data[data.O == obj_index].iterrows():
        accuracy = accuracy_list[inst[1].S]
        v = inst[1].V
        a *= get_factor(data, accuracy, v, possible_values[0], p_v_obj)
        b *= get_factor(data, accuracy, v, possible_values[1], p_v_obj)
    likelihood = [a/(a+b), b/(a+b)]
    return likelihood


def get_accuracy(data, prob, s_index):
    p_sum = 0.
    size = 0.
    for obj_index in sorted(data.O.drop_duplicates()):
        observed_val = list(data[(data.S == s_index) & (data.O == obj_index)].V)
        if len(observed_val) != 0:
            observed_val = observed_val[0]
        else:
            continue
        p_sum += prob[obj_index][observed_val]
        size += 1
    accuracy = p_sum/size
    return accuracy


def get_dist_metric(data, truth_obj_list, prob):
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


def gibbs_sampl(data, accuracy_data, truth_obj_list):
    dist_list = []
    iter_number_list = []

    for t in range(10):
        observ_val, var_index, accuracy_list, s_number = init_var(data=data, accuracy=accuracy_data)
        prob = get_init_prob(data=data)
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
                        prob_old = copy.deepcopy(prob)
                        o_ind = indexes[0].pop()
                        prob[o_ind] = get_prob(data=data, accuracy_list=accuracy_list, obj_index=o_ind, prob_old=prob_old)
                    else:
                        s_index = indexes[1].pop()
                        accuracy_list[s_index] = get_accuracy(data=data, prob=prob, s_index=s_index)
                elif len(indexes[0])==0 and len(indexes[1])!=0:
                        s_index = indexes[1].pop()
                        accuracy_list[s_index] = get_accuracy(data=data, prob=prob, s_index=s_index)
                elif len(indexes[0])!=0 and len(indexes[1])==0:
                    prob_old = copy.deepcopy(prob)
                    o_ind = indexes[0].pop()
                    prob[o_ind] = get_prob(data=data, accuracy_list=accuracy_list, obj_index=o_ind, prob_old=prob_old)
                else:
                    round_compl = True
            iter_number += 1
            accuracy_delta = max([abs(k-l) for k, l in zip(accuracy_prev, accuracy_list)])
        dist_metric = get_dist_metric(data=data, truth_obj_list=truth_obj_list, prob=prob)
        dist_list.append(dist_metric)
        iter_number_list.append(iter_number)
    return [np.mean(dist_list), np.mean(iter_number_list)]
