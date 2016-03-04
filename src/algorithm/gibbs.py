"""
Gibbs sampling truth finder

@author: Evgeny Krivosheev (e.krivoshe@gmail.com)
"""

import pandas as pd
import copy
import random

max_rounds = 30
eps = 0.001


def init_var(data, accuracy):
    observ_val = []
    init_prob = []
    s_number = len(accuracy.S)
    accuracy_list = list(accuracy.A)
    accuracy_ind = sorted(data.S.drop_duplicates())
    obj_index_list = sorted(data.O.drop_duplicates())
    for obj_index in obj_index_list:
        possible_values = sorted(list(set(data[data.O == obj_index].V)))
        observ_val.append(possible_values)
        l = len(possible_values)
        init_prob.append([1./l]*l)
    random.shuffle(obj_index_list)
    random.shuffle(accuracy_ind)
    var_index = [obj_index_list, accuracy_ind]

    return observ_val, init_prob, var_index, accuracy_list, s_number


def get_n_params(data):
    n_list = []
    obj_list = sorted(data.O.drop_duplicates())
    for i in obj_list:
        n = len(data[data.O == i].V.drop_duplicates()) - 1
        n_list.append(n)
    return n_list


def get_prob(data, n, accuracy_list, obj_index):
    likelihood = []
    observed_values = list(data[data.O == obj_index].V)
    possible_values = sorted(list(set(observed_values)))
    if n == 0:
        likelihood.append(1.)
        return likelihood
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
        likelihood.append(p)
    return likelihood


def get_accuracy(data, prob, s_index):
    p_sum = 0.
    size = 0.
    for obj_index in sorted(data.O.drop_duplicates()):
        observed_val = list(data[(data.S == s_index) & (data.O == obj_index)].V)
        if len(observed_val) != 0:
            observed_val == observed_val[0]
        else:
            continue
        size += 1
        possible_values = sorted(list(set(data[data.O == obj_index].V)))
        for v_ind, v in enumerate(possible_values):
            if v == observed_val:
                p_sum += prob[obj_index][v_ind]
                break
    accuracy = p_sum/size
    return accuracy


if __name__ == '__main__':
    data = pd.read_csv('../../data/observation.csv', names=['S', 'O', 'V'])
    accuracy_data = pd.read_csv('../../data/accuracy.csv', names=['S', 'A'])
    truth_obj_list = [6, 8, 9, 15, 16, 10, 11, 7, 18, 20]

    observ_val, prob, var_index, accuracy_list, s_number = init_var(data=data, accuracy=accuracy_data)
    accuracy_list__old = copy.copy(accuracy_list)
    prob_old = copy.deepcopy(prob)
    n_list = get_n_params(data=data)

    possible_values = []
    for obj_index in sorted(data.O.drop_duplicates()):
        val = sorted(list(set(data[data.O == obj_index].V)))
        possible_values.append(val)

    accuracy_delta = 0.3
    iter_number = 0
    while accuracy_delta > eps and iter_number < max_rounds:
        accuracy_prev = copy.copy(accuracy_list)
        round_compl = False
        while not round_compl:
            if len(var_index[0])!= 0 and len(var_index[1])!= 0:
                r = random.randint(0, 1)
                if r == 1:
                    o_ind = var_index[0].pop()
                    prob[o_ind] = get_prob(data=data, n=n_list[o_ind], accuracy_list=accuracy_list, obj_index=o_ind)
                else:
                    s_index = var_index[1].pop()
                    accuracy_list[s_index] = get_accuracy(data=data, prob=prob, s_index=s_index)
            elif len(var_index[0])==0 and len(var_index[1])!=0:
                    s_index = var_index[1].pop()
                    accuracy_list[s_index] = get_accuracy(data=data, prob=prob, s_index=s_index)
            elif len(var_index[0])!=0 and len(var_index[1])==0:
                o_ind = var_index[0].pop()
                prob[o_ind] = get_prob(data=data, n=n_list[o_ind], accuracy_list=accuracy_list, obj_index=o_ind)
            else:
                round_compl = True
        iter_number += 1
        accuracy_delta = max([abs(k-l) for k, l in zip(accuracy_prev, accuracy_list)])
    print accuracy_list
    print iter_number
