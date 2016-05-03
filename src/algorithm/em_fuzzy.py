import numpy as np
import pandas as pd
import random
import copy
import itertools


max_rounds = 300
eps = 10e-5
possible_values = [0, 1]


def init_var(data):
    init_prob = []
    s_number = len(data.S.drop_duplicates())
    accuracy_list = [random.uniform(0.6, 0.95) for i in range(s_number)]
    l = len(possible_values)
    obj_index_list = sorted(data.O.drop_duplicates())
    for obj_index in obj_index_list:
        init_prob.append([1./l]*l)
    number_of_cl = len(obj_index_list)/2
    cl_list = range(number_of_cl)
    pi_list = [random.uniform(0.7, 1)]*number_of_cl

    return [init_prob, accuracy_list, cl_list, pi_list]


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


def e_step(data, accuracy_list, pi_list, cl_list):
    cluster_tables = []
    for cl_ind in cl_list:
        cl_obj = [cl_ind*2, cl_ind*2 + 1]
        psi_cl = data[data.O.isin(cl_obj)]
        cl_set_list = [p for p in itertools.product(possible_values, repeat=len(psi_cl)+2)]
        for set_ind, cl_set in enumerate(cl_set_list):
            pr = 1
            for g_ind, psi in enumerate(psi_cl.iterrows()):
                psi = psi[1]
                accuracy = accuracy_list[psi.S]
                g_val = cl_set[g_ind+2]
                if g_val == 1:
                    pr_item = pi_list[cl_ind]
                    pr_item *= accuracy if psi.V == cl_set[psi.O % 2] else 1 - accuracy
                else:
                    pr_item = 1 - pi_list[cl_ind]
                    obj_ind = 1 if psi.O % 2 == 0 else 0
                    pr_item *= accuracy if psi.V == cl_set[obj_ind % 2] else 1 - accuracy
                pr *= pr_item
            cl_set_list[set_ind] = cl_set_list[set_ind] + (pr,)
        columns = ['O0_v', 'O1_v'] + list(psi_cl.index) + ['P']
        cl_table = pd.DataFrame(data=cl_set_list, columns=columns)
        cl_table.iloc[:, -1] = cl_table.iloc[:, -1].div(sum(cl_table.iloc[:, -1]))
        cluster_tables.append(cl_table)

    return cluster_tables


def m_step(data, cluster_tables, s_number, cl_list):
    accuracy_list_new = []
    for s in range(s_number):
        s_data = data[data.S == s]
        accuracy = 0.
        for psi in s_data.iterrows():
            g_ind = psi[0]
            psi = psi[1]
            cl_ind = psi.O / 2
            t = cluster_tables[cl_ind]
            if psi.O % 2 == 0:
                obj_g1_ind = 'O0_v'
                obj_g0_ind = 'O1_v'
            else:
                obj_g1_ind = 'O1_v'
                obj_g0_ind = 'O0_v'
            prob = sum(t[((t[g_ind] == 1) & (t[obj_g1_ind] == psi.V)) | ((t[g_ind] == 0) & (t[obj_g0_ind] == psi.V))].P)
            accuracy += prob
        accuracy /= len(s_data)
        accuracy_list_new.append(accuracy)

    pi_list_new = []
    for cl_ind in cl_list:
        cl_obj = [cl_ind*2, cl_ind*2 + 1]
        g_ind_list = data[data.O.isin(cl_obj)].index
        t = cluster_tables[cl_ind]
        pi = 0.
        for g_ind in g_ind_list:
            pi += sum(t[t[g_ind] == 1].P)
        pi /= len(g_ind_list)
        pi_list_new.append(pi)

    return [accuracy_list_new, pi_list_new]


def get_prob(cluster_tables):
    prob = []
    for cl_ind, t in enumerate(cluster_tables):
        prob.append([sum(t[t['O0_v'] == 0].P), sum(t[t['O0_v'] == 1].P)])
        prob.append([sum(t[t['O1_v'] == 0].P), sum(t[t['O1_v'] == 1].P)])

    return prob


def em_fuzzy(data, truth_obj_list):
    dist_list = []
    iter_list = []
    pi_prob_all = []
    accuracy_all = []
    for round in range(1):
        prob, accuracy_list, cl_list, pi_list = init_var(data=data)
        s_number = len(data.S.drop_duplicates())
        accuracy_delta = 0.3
        iter_number = 0
        while accuracy_delta > eps and iter_number < max_rounds:
            cluster_tables = e_step(data=data, accuracy_list=accuracy_list,
                                    pi_list=pi_list, cl_list=cl_list)
            accuracy_prev = copy.copy(accuracy_list)
            accuracy_list, pi_list = m_step(data=data, cluster_tables=cluster_tables,
                                            s_number=s_number, cl_list=cl_list)
            accuracy_delta = max([abs(k-l) for k, l in zip(accuracy_prev, accuracy_list)])
            iter_number += 1

        prob = get_prob(cluster_tables=cluster_tables)
        dist_metric = get_dist_metric(data=data, truth_obj_list=truth_obj_list, prob=prob)
        dist_list.append(dist_metric)
        iter_list.append(iter_number)
        accuracy_all.append(accuracy_list)
        pi_prob_all.append(pi_list)

    accuracy_mean = []
    accuracy_df = pd.DataFrame(data=accuracy_all)
    for s in range(len(accuracy_list)):
        accuracy_mean.append(np.mean(accuracy_df[s]))
    pi_df = pd.DataFrame(data=pi_prob_all)
    pi_mean = []
    for pi in range(len(pi_list)):
        pi_mean.append(np.mean(pi_df[pi]))

    return [np.mean(dist_list), np.mean(iter_list), accuracy_mean, pi_mean]
