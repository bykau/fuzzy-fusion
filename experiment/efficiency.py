import sys
import random
import time
import pandas as pd
import numpy as np
# sys.path.append('/home/evgeny/fuzzy-fusion/src/')
# sys.path.append('/home/evgeny/fuzzy-fusion/experiment/')
sys.path.append('/Users/Evgeny/wonderful_programming/fuzzy-fusion-venv/fuzzy-fusion/src/')
sys.path.append('/Users/Evgeny/wonderful_programming/fuzzy-fusion-venv/fuzzy-fusion/experiment/')
from generator.generator import generator
from algorithm.gibbs_fuzzy import gibbs_fuzzy
from algorithm.gibbs import gibbs
from algorithm.em import em
from algorithm.em_fuzzy import em_fuzzy
from algorithm.m_voting import m_voting


s_number = 10
obj_number_list = [50, 100, 500, 1000, 3000, 5000, 7000, 10000, 12000, 15000]
emf_bj_number_list = [10, 20, 50, 100, 150, 200]
cl_size = 2
possible_values = [0, 1]
cov_list = [0.7]*s_number
p_list = [0.7]*s_number
pi = 0.8


def get_dist(gt, output):
    dist_metric = np.dot(gt, output)
    dist_metric_norm = dist_metric/len(gt)

    return dist_metric_norm


def ef_test():
    print 'pi: {}'.format(pi)
    print '*****'

    pd.DataFrame(data=[], columns=['pi', 's', 'obj_numb', 'mv', 'em', 'g', 'gf'])\
        .to_csv('outputs/efficiency.csv', index_label=False)

    for obj_number in obj_number_list:
        print 'obj number: {}'.format(obj_number)

        ground_truth = [random.randint(0, len(possible_values)-1) for i in range(obj_number)]
        data, g_data = generator(cov_list, p_list, ground_truth, cl_size, pi, possible_values)

        t_mv = time.time()
        mv = m_voting(data=data, truth_obj_list=ground_truth)
        ex_t_mv = time.time() - t_mv
        print 'm_v: {}, seconds: {}'.format(mv, ex_t_mv)

        t_em = time.time()
        em_d, em_it, accuracy_em = em(data=data, truth_obj_list=ground_truth, values=possible_values)
        ex_t_em = time.time() - t_em
        print 'em: {}, seconds: {}'.format(em_d, ex_t_em)

        t_g = time.time()
        g_d, g_it, accuracy_g = gibbs(data=data, truth_obj_list=ground_truth)
        ex_t_g = time.time() - t_g
        print 'g: {}, seconds: {}'.format(g_d, ex_t_g)

        t_gf = time.time()
        gf_d, gf_it, accuracy_gf, pi_gf = gibbs_fuzzy(data=data, truth_obj_list=ground_truth)
        ex_t_gf = time.time() - t_gf
        print 'gf: {}, seconds: {}'.format(gf_d, ex_t_gf)

        print '---'

        result_list = [pi, s_number, obj_number, ex_t_mv, ex_t_em, ex_t_g, ex_t_gf]
        data_frame = pd.read_csv('outputs/efficiency.csv')
        data_frame = data_frame.append(pd.DataFrame(data=[result_list],
                                                    columns=['pi', 's', 'obj_numb',
                                                             'mv', 'em', 'g', 'gf']))
        data_frame.to_csv('outputs/efficiency.csv', index_label=False)


def ef_test_emf():
    print 'pi: {}'.format(pi)
    print '*****'

    pd.DataFrame(data=[], columns=['pi', 's', 'obj_numb', 'emf'])\
        .to_csv('outputs/efficiency_emf.csv', index_label=False)

    for obj_number in emf_bj_number_list:

        print 'obj number: {}'.format(obj_number)

        ground_truth = [random.randint(0, len(possible_values)-1) for i in range(obj_number)]
        data, g_data = generator(cov_list, p_list, ground_truth, cl_size, pi, possible_values)

        emf_t = time.time()
        em_f, em_f_it, accuracy_em_f, pi_em_f = em_fuzzy(data=data, truth_obj_list=ground_truth)
        ex_emf = time.time() - emf_t
        print 'em_f: {}, seconds: {}'.format(em_f, ex_emf)
        print '---'

        result_list = [pi, s_number, obj_number, ex_emf]
        data_frame = pd.read_csv('outputs/efficiency_emf.csv')
        data_frame = data_frame.append(pd.DataFrame(data=[result_list],
                                                    columns=['pi', 's', 'obj_numb', 'emf']))
        data_frame.to_csv('outputs/efficiency_emf.csv', index_label=False)


if __name__ == '__main__':
    ef_test()
    # ef_test_emf()
