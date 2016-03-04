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
    obj_index_list = sorted(data.O.drop_duplicates())
    for obj_index in obj_index_list:
        possible_values = sorted(list(set(data[data.O == obj_index].V)))
        observ_val.append(possible_values)
        l = len(possible_values)
        init_prob.append([1./l]*l)
    random.shuffle(obj_index_list)
    random.shuffle(accuracy_list)
    var_index = [obj_index_list, accuracy_list]

    return observ_val, init_prob, var_index, accuracy_list, s_number


if __name__ == '__main__':
    data = pd.read_csv('../../data/observation.csv', names=['S', 'O', 'V'])
    accuracy_data = pd.read_csv('../../data/accuracy.csv', names=['S', 'A'])
    truth_obj_list = [6, 8, 9, 15, 16, 10, 11, 7, 18, 20]

    observ_val, prob, var_index, accuracy, s_number = init_var(data=data, accuracy=accuracy_data)
    accuracy_old = copy.copy(accuracy)
    prob_old = copy.deepcopy(prob)

    r = random.randint(0, 1)
    if r == 1:
        o_ind = var_index[0].pop()
    else:
        a_index = var_index[1].pop()
