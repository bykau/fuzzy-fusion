import numpy as np
import random
import copy
from scipy.stats import beta

max_rounds = 300
eps = 10e-5
possible_values = [0, 1]
# accuracy hyperparameters
# mean(A) = 0.7, std^2= 0.1
# alpha1, alpha2 = 77, 33
alpha1, alpha2 = 1, 1
beta1, beta2 = 1, 1
gamma1, gamma2 = 1, 1


def init_var(data):
    s_ind = sorted(data.S.drop_duplicates())
    obj_index_list = sorted(data.O.drop_duplicates())
    var_index = [obj_index_list, s_ind]
    # accuracy_list = beta.rvs(alpha1, alpha2, size=len(s_ind))
    accuracy_list = [random.uniform(0.6, 0.95) for i in range(s_number)]
    # accuracy_list = beta.rvs(alpha1, alpha2, size=s_number)
    pi_init = beta.rvs(gamma1, gamma2, size=1)
    g_values = np.random.binomial(1, pi_init, len(data))
    # g_values = [random.choice(possible_values) for i in range(len(data))]
    obj_values = [random.choice(possible_values) for i in range(len(ground_truth))]
    pi_prob = [0.5]*(len(obj_index_list)/2)
    # random.shuffle(obj_index_list)
    # random.shuffle(accuracy_ind)

    init_prob = []
    l = len(possible_values)
    for obj_index in obj_index_list:
        init_prob.append([1./l]*l)

    counts = []
    for s in s_ind:
        psi_ind_list = []
        counts_list = []
        for psi in data[data.S == s].iterrows():
            psi_ind = psi[0]
            psi_ind_list.append(psi_ind)
            psi = psi[1]
            if g_values[psi_ind] == 1:
                if psi.V == obj_values[psi.O]:
                    counts_list.append(1)
                else:
                    counts_list.append(0)
            else:
                if psi.O % 2 == 0:
                    if psi.V == obj_values[psi.O+1]:
                        counts_list.append(1)
                    else:
                        counts_list.append(0)
                else:
                    if psi.V == obj_values[psi.O-1]:
                        counts_list.append(1)
                    else:
                        counts_list.append(0)
        counts.append(pd.DataFrame(counts_list, columns=['c']).set_index([psi_ind_list]))

    return [var_index, g_values, obj_values, counts, init_prob, pi_prob, accuracy_list]


def get_dist_metric(data, truth_obj_list, prob):
    prob_gt = []
    val = []
    l = len(possible_values)
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


def get_o(o_ind, g_values, obj_values, counts, data, prob, accuracy_list):
    l_p = []
    l_c = []
    if o_ind % 2 == 0:
        cl = [o_ind, o_ind+1]
    else:
        cl = [o_ind-1, o_ind]
    psi_cl = data[data.O.isin(cl)]
    s_in_cluster = list(psi_cl.S.drop_duplicates())
    for v in possible_values:
        counts_v = copy.deepcopy(counts)
        pr_v = prob[o_ind][v]
        for s in s_in_cluster:
            flag = True
            for psi in psi_cl[psi_cl.S == s].iterrows():
                psi_ind = psi[0]
                psi = psi[1]
                c_old = counts_v[s].at[psi_ind, 'c']
                if g_values[psi_ind] == 1 and psi.O == o_ind:
                    c_new = 1 if psi.V == v else 0
                    if c_new != c_old:
                        counts_v[s].at[psi_ind, 'c'] = c_new
                elif psi.O != o_ind and g_values[psi_ind] == 0:
                    c_new = 1 if psi.V == v else 0
                    if c_new != c_old:
                        counts_v[s].at[psi_ind, 'c'] = c_new
                else:
                    flag = False
            if flag:
                accuracy = accuracy_list[s]
                n_true = len(counts_v[s][counts_v[s].c == 1])
                n_false = len(counts_v[s][counts_v[s].c == 0])
                pr_v *= accuracy**n_true*(1-accuracy)**n_false
        l_p.append(pr_v)
        l_c.append(counts_v)
    norm_const = sum(l_p)
    l_p[0] /=norm_const
    l_p[1] /=norm_const
    v_new = np.random.binomial(1, l_p[1], 1)[0]
    counts_new = l_c[v_new]

    return [v_new, counts_new, l_p]


def get_g(g_ind, g_prev, pi_prob, obj_values, accuracy_list, counts, data):
    l_p = []
    psi = data.iloc[g_ind]
    s = psi.S
    accuracy = accuracy_list[s]
    cluster = psi.O/2
    for g in possible_values:
        pr_pi = pi_prob[cluster]**g*(1-pi_prob[cluster])**(1-g)
        if g == 1:
            if psi.V == obj_values[psi.O]:
                pr_pi *= accuracy
            else:
                pr_pi *= 1-accuracy
        else:
            if psi.O % 2 == 0:
                if psi.V == obj_values[psi.O+1]:
                    pr_pi *= accuracy
                else:
                    pr_pi *= 1-accuracy
            else:
                if psi.V == obj_values[psi.O-1]:
                    pr_pi *= accuracy
                else:
                    pr_pi *= 1-accuracy
        l_p.append(pr_pi)
    norm_const = sum(l_p)
    l_p[0] /=norm_const
    l_p[1] /=norm_const
    g_new = np.random.binomial(1, l_p[1], 1)[0]
    if g_new != g_prev:
        if g_new == 1:
            if psi.V == obj_values[psi.O]:
                counts[s].at[g_ind, 'c'] = 1
            else:
                counts[s].at[g_ind, 'c'] = 0
        else:
            if psi.O % 2 == 0:
                if psi.V == obj_values[psi.O+1]:
                    counts[s].at[g_ind, 'c'] = 1
                else:
                    counts[s].at[g_ind, 'c'] = 0
            else:
                if psi.V == obj_values[psi.O-1]:
                    counts[s].at[g_ind, 'c'] = 1
                else:
                    counts[s].at[g_ind, 'c'] = 0

    return g_new


def get_pi(cl_ind, g_values, data):
    cl = [cl_ind*2, cl_ind*2+1]
    psi_cl_list = list(data[data.O.isin(cl)].index)
    count_p = 0
    count_m = 0
    for g_ind in psi_cl_list:
        g = g_values[g_ind]
        if g == 1:
            count_p += 1
        else:
            count_m += 1
    pi_new = beta.rvs(count_p + gamma1, count_m + gamma2, size=1)

    return pi_new


def get_a(s_counts):
    count_p = len(s_counts[s_counts.c == 1])
    count_m = len(s_counts[s_counts.c == 0])
    a_new = beta.rvs(count_p + alpha1, count_m + alpha2, size=1)

    return a_new


def gibbs_fuzzy(data, accuracy_data, g_data, truth_obj_list):
    dist_list = []
    iter_list = []
    for round in range(10):
        var_index, g_values, obj_values, counts, prob, pi_prob, accuracy_list = init_var(data=data)
        iter_number = 0
        dist_metric = 1.
        dist_delta = 0.3
        dist_temp = []
        while dist_delta > eps and iter_number < max_rounds:
            accuracy_prev = copy.copy(accuracy_list)
            for o_ind in var_index[0]:
                 obj_values[o_ind], counts, prob[o_ind] = get_o(o_ind=o_ind, g_values=g_values,
                                                                obj_values=obj_values, counts=counts,
                                                                data=data, prob=prob, accuracy_list=accuracy_list)

            for g_ind in range(len(g_values)):
                g_prev = g_values[g_ind]
                g_values[g_ind] = get_g(g_ind=g_ind, g_prev=g_prev, pi_prob=pi_prob, obj_values=obj_values,
                                        accuracy_list=accuracy_list, counts=counts, data=data)

            for cl_ind in range(len(pi_prob)):
                pi_prob[cl_ind] = get_pi(cl_ind=cl_ind, g_values=g_values, data=data)

            for s_ind in range(len(accuracy_list)):
                s_counts = counts[s_ind]
                accuracy_list[s_ind] = get_a(s_counts=s_counts)
            iter_number += 1
            dist_metric_old = dist_metric
            dist_metric = get_dist_metric(data=data, truth_obj_list=truth_obj_list, prob=prob)
            dist_delta = abs(dist_metric-dist_metric_old)
            # print 'dist: {}'.format(dist_metric)
            dist_temp.append(dist_metric)
        #     dist_delta = abs(dist_metric-dist_metric_old)
            # print dist_delta
        print iter_number

        if len(dist_temp) >= 20:
            dist_metric = np.mean(dist_temp[19:])
        else:
            dist_metric = dist_temp[-1]
        dist_list.append(dist_metric)
        iter_list.append(iter_number)
        print 'dist: {}'.format(dist_metric)
        print '------'
        # print '------'
        # print 'dist: {}'.format(dist_metric)
    # print 'std: {}'.format(np.std(dist_metric))
    # print dist_list
    return [np.mean(dist_list), np.mean(iter_list)]






import sys
import time
import pandas as pd
# sys.path.append('/home/evgeny/fuzzy-fusion/src/')
sys.path.append('/Users/Evgeny/wonderful_programming/fuzzy-fusion-venv/fuzzy-fusion/src/')
from generator.generator import generator
from algorithm.gibbs import gibbs_sampl
from algorithm.em import em

print 'Python version ' + sys.version
print 'Pandas version ' + pd.__version__

s_number = 5
obj_number = 20
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
#         ground_truth.append(c_g.csv)
#     else:
#         ground_truth.append(0)
# possible_values = range(2)
for g_true in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]:

    print g_true
    print '*****'

    for i in range(10):
        print i
        ground_truth = [random.randint(0, len(possible_values)-1) for i in range(obj_number)]
        data, g_data = generator(cov_list, p_list, ground_truth, cl_size, g_true, possible_values)

        # t_em = time.time()
        em_d, em_it = em(data=data, truth_obj_list=ground_truth, values=possible_values)
        print 'em: {}'.format(em_d)

        # ex_t_em = time.time() - t_em
        # em_t.append(ex_t_em)
#         print("--- %s seconds em ---" % (ex_t_em))

        while True:
            try:
                # t_g = time.time()
                # g_d, g_it = gibbs_sampl(data=data, truth_obj_list=ground_truth, values=possible_values)
                # print 'g: {}'.format(g_d)
                # ex_t_g = time.time() - t_g
                # g_t.append(ex_t_g)
                # print("--- %s seconds g ---" % (ex_t_g))

                # t_gf = time.time()
                gf_d, gf_it = gibbs_fuzzy(data=data, accuracy_data=accuracy_data, g_data=g_data,
                                          truth_obj_list=ground_truth)
                print 'gf: {}'.format(gf_d)
                # print gf_it
                print '---'
                # ex_t_gf = time.time() - t_gf
                # gf_t.append(ex_t_gf)
#                 print("--- %s seconds gf ---" % (ex_t_gf))
                break
            except ZeroDivisionError:
                print 'zero {}'.format(i)
        result_list.append([g_true, em_d, gf_d])
df = pd.DataFrame(data=result_list, columns=['g_true', 'em', 'gf'])
df.to_csv('par_alpha.csv')
