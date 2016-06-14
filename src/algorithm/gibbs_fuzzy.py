import numpy as np
import random
import copy
from scipy.stats import beta
import pandas as pd


max_rounds = 5
alpha1, alpha2 = 1, 1
beta1, beta2 = 1, 1
gamma1, gamma2 = 1, 1


def init_var(data, s_number):
    s_index_list = range(s_number)
    obj_index_list = data.keys()
    var_index = [obj_index_list, s_index_list]
    accuracy_list = [random.uniform(0.8, 0.95) for i in range(s_number)]
    pi_init = 0.8

    pi_prob = dict(zip(range(len(obj_index_list)/2), [pi_init]*(len(obj_index_list)/2)))
    g_values = {}
    for obj_index, values in data.iteritems():
        g_values.update({obj_index: [values[0], [1]*len(values[1])]})

    obj_values = {}
    init_prob = {}
    counts = {}
    for obj_index in obj_index_list:
        # possible_values, values = get_possible_values(obj_index=obj_index, data=data, g_values=g_values)
        values = data[obj_index][1]
        possible_values = sorted(set(values))
        obj_val = max(set(values), key=values.count)
        obj_values.update({obj_index: obj_val})
        l = len(possible_values)
        init_prob.update({obj_index: [1./l]*l})

        sources = data[obj_index][0]
        counts_list = []
        for val in values:
            if val == obj_val:
                counts_list.append(1)
            else:
                counts_list.append(0)
        counts.update({obj_index: [sources, counts_list]})

    return [var_index, g_values, obj_values, counts, init_prob, pi_prob, accuracy_list]


def update_obj(obj_index, g_values, obj_values, counts, data, accuracy_list, prob):
    l_p = []
    if obj_index % 2 == 0:
        cl = [obj_index, obj_index+1]
    else:
        cl = [obj_index-1, obj_index]
    possible_values = get_possible_values(obj_index=obj_index, data=data, g_values=g_values)[0]

    n = len(possible_values) - 1
    for v_ind, v in enumerate(possible_values):
        l_p.append(1.)
        for obj in cl:
            obj_s = data[obj][0]
            psi_list = data[obj][1]
            for s, psi_val in zip(obj_s, psi_list):
                accuracy = accuracy_list[s]
                psi_ind = psi_list.index(psi_val)
                g = g_values[obj][1][psi_ind]
                if (obj == obj_index and g == 1) or (obj != obj_index and g == 0):
                    if psi_val == v:
                        l_p[v_ind] *= accuracy
                    else:
                        l_p[v_ind] *= (1-accuracy)/n
    norm_const = sum(l_p)
    for v_ind in range(len(possible_values)):
        l_p[v_ind] /= norm_const
    mult_trial = list(np.random.multinomial(1, l_p, size=1)[0])
    v_new_ind = mult_trial.index(1)
    v_new = possible_values[v_new_ind]
    obj_values.update({obj_index: v_new})
    prob.update({obj_index: l_p})

    for obj in cl:
        obj_s = data[obj][0]
        psi_list = data[obj][1]
        for s, psi_val in zip(obj_s, psi_list):
            psi_ind = psi_list.index(psi_val)
            g = g_values[obj][1][psi_ind]
            if (obj == obj_index and g == 1) or (obj != obj_index and g == 0):
                c_prev = counts[obj][1][psi_ind]
                c_new = 1 if psi_val == v_new else 0
                if c_new != c_prev:
                    counts[obj][1][psi_ind] = c_new


def update_g(s, obj_index, g_values, pi_prob, obj_values, accuracy, counts, data):
    l_p = []
    possible_values = [0, 1]
    cluster = obj_index/2
    psi_index = data[obj_index][0].index(s)
    psi_val = data[obj_index][1][psi_index]
    g_prev = g_values[obj_index][1][psi_index]
    for g in possible_values:
        pr_pi = pi_prob[cluster]**g*(1-pi_prob[cluster])**(1-g)
        if g == 1:
            g_values[obj_index][1][psi_index] = 1
            n = len(get_possible_values(obj_index=obj_index, data=data, g_values=g_values)[0]) - 1
            if psi_val == obj_values[obj_index]:
                pr_pi *= accuracy
            else:
                if n == 0:
                    pr_pi *= 0.
                else:
                    pr_pi *= (1-accuracy)/n
        else:
            g_values[obj_index][1][psi_index] = 0
            if obj_index % 2 == 0:
                n = len(get_possible_values(obj_index=obj_index+1, data=data, g_values=g_values)[0]) - 1
                if psi_val == obj_values[obj_index+1]:
                    pr_pi *= accuracy
                else:
                    if n == 0:
                        pr_pi *= 0.
                    else:
                        pr_pi *= (1-accuracy)/n
            else:
                n = len(get_possible_values(obj_index=obj_index-1, data=data, g_values=g_values)[0]) - 1
                if psi_val == obj_values[obj_index-1]:
                    pr_pi *= accuracy
                else:
                    if n == 0:
                        pr_pi *= 0.
                    else:
                        pr_pi *= (1-accuracy)/n
        l_p.append(pr_pi)
    norm_const = sum(l_p)
    l_p[0] /= norm_const
    l_p[1] /= norm_const
    g_new = np.random.binomial(1, l_p[1], 1)[0]

    if g_new != g_prev:
        g_values[obj_index][1][psi_index] = g_new

        if g_new == 1:
            if psi_val == obj_values[obj_index]:
                counts[obj_index][1][psi_index] = 1
            else:
                counts[obj_index][1][psi_index] = 0
        else:
            if obj_index % 2 == 0:
                if psi_val == obj_values[obj_index+1]:
                    counts[obj_index][1][psi_index] = 1
                else:
                    counts[obj_index][1][psi_index] = 0
            else:
                if psi_val == obj_values[obj_index-1]:
                    counts[obj_index][1][psi_index] = 1
                else:
                    counts[obj_index][1][psi_index] = 0


def get_pi(cl_ind, g_values, data):
    cl = [cl_ind*2, cl_ind*2+1]
    count_p, count_m = 0, 0
    for obj in cl:
        for g in g_values[obj][1]:
            if g == 1:
                count_p += 1
            else:
                count_m += 1
    pi_new = beta.rvs(count_p + gamma1, count_m + gamma2, size=1)[0]

    return pi_new


def get_a(s_counts):
    count_p = len(s_counts[s_counts.c == 1])
    count_m = len(s_counts[s_counts.c == 0])
    a_new = beta.rvs(count_p + alpha1, count_m + alpha2, size=1)[0]

    return a_new


def get_possible_values(obj_index, data, g_values):
    if obj_index % 2 == 0:
        cl = [obj_index, obj_index+1]
    else:
        cl = [obj_index-1, obj_index]
    possible_values = []
    values = []
    for obj in cl:
        obj_obs = data[obj][1]
        obj_g_values = g_values[obj][1]
        for val, g in zip(obj_obs, obj_g_values):
            if (g == 1 and obj == obj_index) \
                    or (g == 0 and obj != obj_index):
                if val not in possible_values:
                    possible_values.append(val)
                values.append(val)
    possible_values = sorted(possible_values)

    return [possible_values, values]


def get_dist_metric(data, truth_obj_list, prob, g_values):
    prob_gt = []
    val = []
    for obj_index in range(len(truth_obj_list)):
        if obj_index % 2 == 0:
            cl = [obj_index, obj_index+1]
        else:
            cl = [obj_index-1, obj_index]
        psi_cl = data[data.O.isin(cl)]
        possible_values = []
        for psi in psi_cl.iterrows():
            psi_ind = psi[0]
            psi = psi[1]
            if (g_values[psi_ind] == 1 and psi.O == obj_index)\
                    or (g_values[psi_ind] == 0 and psi.O != obj_index):
                if psi.V not in possible_values:
                    possible_values.append(psi.V)
        possible_values = sorted(possible_values)
        val.append(possible_values)
        l = len(possible_values)
        prob_gt.append([0]*l)

    for obj_ind, v_true in enumerate(truth_obj_list):
        for v_ind, v in enumerate(val[obj_ind]):
            if v == v_true:
                prob_gt[obj_ind][v_ind] = 1.
    prob_gt_vector = []
    prob_vector = []
    for i in range(len(prob_gt)):
        prob_gt_vector += prob_gt[i]
        prob_vector += prob[i]
    dist_metric = np.dot(prob_gt_vector, prob_vector)
    dist_metric_norm = dist_metric/len(prob_gt)

    return dist_metric_norm


def gibbs_fuzzy(data=None, gt=None, accuracy_truth=None, s_number=None):
    dist_list = []
    iter_list = []
    pi_prob_all = []
    accuracy_all = []
    for round in range(1):
        var_index, g_values, obj_values, counts, prob, pi_prob, accuracy_list = init_var(data=data, s_number=s_number)
        iter_number = 0
        dist_temp = []
        precision_temp = []
        while iter_number < max_rounds:
            for obj_index, values in g_values.iteritems():
                for s in values[0]:
                    accuracy = accuracy_list[s]
                    update_g(s=s, obj_index=obj_index, g_values=g_values, pi_prob=pi_prob, obj_values=obj_values,
                             accuracy=accuracy, counts=counts, data=data)

            for obj_index in var_index[0]:
                update_obj(obj_index=obj_index, g_values=g_values,
                           obj_values=obj_values, counts=counts,
                           data=data, accuracy_list=accuracy_list, prob=prob)

            for cl_ind in range(len(pi_prob)):
                pi_prob[cl_ind] = get_pi(cl_ind=cl_ind, g_values=g_values, data=data)

            for s_ind in var_index[1]:
                s_counts = counts[s_ind]
                accuracy_list[s_ind] = get_a(s_counts=s_counts)
            iter_number += 1

            dist_metric = get_dist_metric(data=data, truth_obj_list=truth_obj_list,
                                          prob=prob[0:len(truth_obj_list)], g_values=g_values)
            # precision = get_precision(data=data, truth_obj_list=truth_obj_list, prob=prob[0:len(truth_obj_list)])
            # precision_temp.append(precision)
            dist_temp.append(dist_metric)
            # print dist_metric

        pi_prob_all.append(pi_prob)
        accuracy_all.append(accuracy_list)
        dist_metric = np.mean(dist_temp[-3:])
        dist_list.append(dist_metric)
        iter_list.append(iter_number)

    dist_metric = np.mean(dist_temp[-3:])
    # precision = np.mean(precision_temp[-10:])

    accuracy_mean = []
    accuracy_df = pd.DataFrame(data=accuracy_all)
    for s in range(len(accuracy_list)):
        accuracy_mean.append(np.mean(accuracy_df[s]))
    pi_df = pd.DataFrame(data=pi_prob_all)
    pi_mean = []
    for pi in range(len(pi_prob)):
        pi_mean.append(np.mean(pi_df[pi]))

    return [dist_metric, np.mean(iter_list), accuracy_mean, pi_mean]
