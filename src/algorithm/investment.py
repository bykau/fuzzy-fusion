'''
The implementation of Investment algorithm from
Pasternack, J & Roth, D., (2010),
Knowing What to Believe. In COLING.

@author: Evgeny Krivosheev (e.krivoshe@gmail.com)
'''

import random
from common import get_alg_accuracy

max_rounds = 100
eps = 10e-3
g = 1.2


def get_trustw(data, belief, s_obs_count, trustw_prev):
    trustw_new = []
    for s in s_obs_count.keys():
        trustw_s_prev = trustw_prev[s]
        s_trustw_new = 0.
        for obj_index in data.keys():
            obj_data = data[obj_index]
            if s not in obj_data[0]:
                continue
            obj_possible_values = sorted(set(obj_data[1]))
            observed_val = obj_data[1][obj_data[0].index(s)]
            val_ind = obj_possible_values.index(observed_val)
            claim_belief = belief[obj_index][val_ind]

            denom = 0.
            for s_ind in obj_data[0]:
                s_trustw = trustw_prev[s_ind]
                s_count = s_obs_count[s_ind]
                denom += s_trustw/s_count

            s_trustw_new += claim_belief/denom
        s_trustw_new = s_trustw_new*trustw_s_prev/s_obs_count[s]
        trustw_new.append(s_trustw_new)

    t_max = max(trustw_new)
    trustw_new = map(lambda t: t/t_max, trustw_new)

    return trustw_new


def get_belief(data, trustw_list, s_obs_count):
    belief = {}
    for obj_index in data.keys():
        obj_data = data[obj_index]
        possible_values = sorted(set(obj_data[1]))
        l = len(possible_values)
        term_list = [0.]*l
        for s_ind, v in zip(obj_data[0], obj_data[1]):
            s_trustw = trustw_list[s_ind]
            term_ind = possible_values.index(v)
            s_count = s_obs_count[s_ind]
            term_list[term_ind] += s_trustw/s_count

        for term_ind, term in enumerate(term_list[:]):
            term_list[term_ind] = term**g

        b_max = max(term_list)
        if b_max == 0.:
            b_max = 1.
        term_list = map(lambda b: b/b_max, term_list)
        belief.update({obj_index: term_list})

    return belief


def investment(data=None, gt=None, accuracy_truth=None, s_number=None):
    sources = range(s_number)
    s_obs_count = dict(zip(sources, [0]*s_number))
    for obj in data.values():
        for s in obj[0]:
            s_obs_count[s] += 1
    trustw_list = [random.uniform(0.8, 0.95) for i in range(s_number)]
    trustw_delta = 0.3
    iter_number = 0
    while trustw_delta > eps and iter_number < max_rounds:
        belief = get_belief(data=data, trustw_list=trustw_list, s_obs_count=s_obs_count)
        trustw_prev = trustw_list[:]
        trustw_list = get_trustw(data=data, belief=belief, s_obs_count=s_obs_count, trustw_prev=trustw_prev)
        trustw_delta = max([abs(k-l) for k, l in zip(trustw_prev, trustw_list)])
        iter_number += 1
    alg_accuracy = get_alg_accuracy(data=data, gt=gt, belief=belief)

    return alg_accuracy
