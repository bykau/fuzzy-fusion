import numpy as np
import random
import copy

max_rounds = 30
eps = 0.05


def init_var(data, accuracy):
    s_number = len(accuracy.S)
    accuracy_list = list(accuracy.A)
    accuracy_ind = sorted(data.S.drop_duplicates())
    obj_index_list = sorted(data.O.drop_duplicates())
    random.shuffle(obj_index_list)
    random.shuffle(accuracy_ind)
    var_index = [obj_index_list, accuracy_ind]
    return [var_index, accuracy_list, s_number]


def init_prob(data, values):
    init_prob = []
    l = len(values)
    obj_index_list = sorted(data.O.drop_duplicates())
    for obj_index in obj_index_list:
        init_prob.append([1./l]*l)
    return init_prob


def get_factor(psi, G, accuracy_list, obj_index, v, values, n):
    factor = 0.
    for g in G.iterrows():
        g = g[1]
        if g.Oj == obj_index and g.Oi == obj_index and psi.O == obj_index:
            if psi.V == v:
                factor += accuracy_list[psi.S]*g.P
            else:
                factor += (1-accuracy_list[psi.S])*g.P/n
        elif g.Oj != obj_index and g.Oi == obj_index and psi.O == obj_index:
            factor += g.P
        elif g.Oi != obj_index:
            if g.Oj == obj_index:
                if psi.V == v:
                    factor += accuracy_list[psi.S]*g.P
                else:
                    factor += (1-accuracy_list[psi.S])*g.P
            else:
                factor += g.P

    return factor


def get_prob(data, g_data, accuracy_list, obj_index, values):
    prob = []
    G = g_data[g_data.Oi.isin(g_data.Oi[g_data.Oj == obj_index])]
    Psi = data[data.O.isin(G.Oi.drop_duplicates())]
    l = len(values)
    n = l - 1
    term_list = [1]*l
    for psi in Psi.iterrows():
        psi = psi[1]
        g_obj = G[G.Oi == psi.O]
        for v_ind, v in enumerate(values):
            term_list[v_ind] *= get_factor(psi=psi, G=g_obj, accuracy_list=accuracy_list,
                                           obj_index=obj_index, v=v, values=values, n=n)
    denom = sum(term_list)
    for v_ind in range(l):
        prob.append(term_list[v_ind]/denom)
    return prob


def get_accuracy(data, g_data, prob, s_index, values):
    n = len(values) - 1
    Psi_s = data[data.S == s_index]
    G = g_data[g_data.Oi.isin(list(Psi_s.O))]
    size = len(Psi_s)
    p_sum = 0.
    claster_flag = True
    for psi in Psi_s.iterrows():
        psi = psi[1]
        a, b = 0., 0.
        for g in G.iterrows():
            g = g[1]
            if g.Oi != psi.O:
                continue
            if g.P == 1:
                psi_prob = prob[psi.O][psi.V]
                claster_flag = False
            else:
                oj_ind = int(g.Oj)
                a += prob[oj_ind][psi.V]*g.P
                b += (1-prob[oj_ind][psi.V])*g.P/n
        if claster_flag:
            psi_prob = a/(a+b)
        p_sum += psi_prob
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


def gibbs_fuzzy(data, accuracy_data, g_data, truth_obj_list, values):
    var_index, accuracy_list, s_number = init_var(data=data, accuracy=accuracy_data)
    prob = init_prob(data=data, values=values)
    accuracy_delta = 0.3
    iter_number = 0
    while accuracy_delta > eps and iter_number < max_rounds:
        indexes = copy.deepcopy(var_index)
        accuracy_prev = copy.copy(accuracy_list)
        for o_ind in indexes[0]:
            prob[o_ind] = get_prob(data=data, g_data=g_data, accuracy_list=accuracy_list,
                                   obj_index=o_ind, values=values)
        for s_index in indexes[1]:
            accuracy_list[s_index] = get_accuracy(data=data, g_data=g_data, prob=prob, s_index=s_index, values=values)

        iter_number += 1
        accuracy_delta = max([abs(k-l) for k, l in zip(accuracy_prev, accuracy_list)])
    dist_metric = get_dist_metric(data=data, truth_obj_list=truth_obj_list, prob=prob, values=values)
    return [dist_metric, iter_number]






import sys
import time
import pandas as pd
sys.path.append('/Users/Evgeny/wonderful_programming/fuzzy-fusion-venv/fuzzy-fusion/src/')
from generator.generator import generator
from algorithm.gibbs import gibbs_sampl
from algorithm.gibbs_fuzzy import gibbs_fuzzy
from algorithm.em import em

print 'Python version ' + sys.version
print 'Pandas version ' + pd.__version__

s_number = 12
obj_number = 50
cl_size = 2
cov_list = [0.7]*s_number
p_list = [0.7]*s_number
accuracy_list = [random.uniform(0.6, 0.95) for i in range(s_number)]
accuracy_for_df = [[i, accuracy_list[i]] for i in range(s_number)]
accuracy_data = pd.DataFrame(accuracy_for_df, columns=['S', 'A'])

result_list = []
em_t = []
g_t = []
gf_t = []
# ground_truth = []
# for i in range(obj_number):
#     if i % 2 == 0:
#         ground_truth.append(1)
#     else:
#         ground_truth.append(0)
for g_true in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]: #

    print g_true
    print '*****'

    for i in range(3):
        print i
        ground_truth = [random.randint(0, 3) for i in range(obj_number)]
        data, g_data = generator(cov_list, p_list, ground_truth, cl_size, g_true, [0,1,2,3])

        # t_em = time.time()
        em_d, em_it = em(data=data, accuracy=accuracy_data, truth_obj_list=ground_truth, values=[0,1,2,3])
        print em_d
        # ex_t_em = time.time() - t_em
        # em_t.append(ex_t_em)
#         print("--- %s seconds em ---" % (ex_t_em))

        while True:
            try:
                # t_g = time.time()
                # g_d, g_it = gibbs_sampl(data=data, accuracy_data=accuracy_data, truth_obj_list=ground_truth)
                # ex_t_g = time.time() - t_g
                # g_t.append(ex_t_g)
                # print("--- %s seconds g ---" % (ex_t_g))

                t_gf = time.time()
                gf_d, gf_it = gibbs_fuzzy(data=data, accuracy_data=accuracy_data, g_data=g_data,
                                          truth_obj_list=ground_truth, values=[0,1,2,3])
                print gf_d
                print '---'
                ex_t_gf = time.time() - t_gf
                gf_t.append(ex_t_gf)
#                 print("--- %s seconds gf ---" % (ex_t_gf))
                break
            except ZeroDivisionError:
                print 'zero {}'.fprmat(i)
        # result_list.append([g_true, em_d, gf_d])
