import numpy as np
import random
import itertools


max_rounds = 30
eps = 0.001
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
        cl_table = pd.DataFrame(data=cl_set_list)
        cl_table.iloc[:, -1] = cl_table.iloc[:, -1].div(sum(cl_table.iloc[:, -1]))
        cluster_tables.append(cl_table)

    return cluster_tables



def em_fuzzy(data, truth_obj_list):
    prob, accuracy_list, cl_list, pi_list = init_var(data=data)

    e_step(data, accuracy_list, pi_list, cl_list)
    dist_metric = get_dist_metric(data=data, truth_obj_list=truth_obj_list, prob=prob)












# temporary
#
import sys
import time
import pandas as pd
# sys.path.append('/home/evgeny/fuzzy-fusion/src/')
sys.path.append('/Users/Evgeny/wonderful_programming/fuzzy-fusion-venv/fuzzy-fusion/src/')
from generator.generator import generator
from algorithm.gibbs import gibbs_sampl
from algorithm.em import em
from algorithm.m_voting import m_voting

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
for g_true in [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]:

    print g_true
    print '*****'

    for i in range(10):
        print i
        ground_truth = [random.randint(0, len(possible_values)-1) for i in range(obj_number)]
        data, g_data = generator(cov_list, p_list, ground_truth, cl_size, g_true, possible_values)

        m_v = m_voting(data=data, truth_obj_list=ground_truth)
        print 'm_v: {}'.format(m_v)


        # t_em = time.time()
        # em_d, em_it = em(data=data, truth_obj_list=ground_truth, values=possible_values)
        # print 'em: {}'.format(em_d)

        # ex_t_em = time.time() - t_em
        # em_t.append(ex_t_em)
        # print("--- %s seconds em ---" % (ex_t_em))

        em_f, em_f_it = em_fuzzy(data=data, truth_obj_list=ground_truth)

        while True:
            try:
                # t_g = time.time()
                # g_d, g_it = gibbs_sampl(data=data, truth_obj_list=ground_truth, values=possible_values)
                # print 'g: {}'.format(g_d)
                # ex_t_g = time.time() - t_g
                # g_t.append(ex_t_g)
                # print("--- %s seconds g ---" % (ex_t_g))

                # t_gf = time.time()
                # gf_d, gf_it = gibbs_fuzzy(data=data, accuracy_data=accuracy_data, g_data=g_data,
                #                           truth_obj_list=ground_truth)
                # print 'gf: {}'.format(gf_d)
                # print gf_it
                # print '---'
                # ex_t_gf = time.time() - t_gf
                # gf_t.append(ex_t_gf)
                # print("--- %s seconds gf ---" % (ex_t_gf))
                break
            except ZeroDivisionError:
                print 'zero {}'.format(i)
        result_list.append([g_true, m_v, em_d, gf_d])
df = pd.DataFrame(data=result_list, columns=['g_true', 'mv', 'em', 'gf'])
df.to_csv('10.csv')
