import sys
import copy
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
obj_number = 20
cl_size = 2
possible_values = range(2)
cov_list = [1.]*s_number
p_list = [1.]*s_number
pi_list = [1., 0.95, 0.9, 0.85, 0.8, 0.75, 0.7]


def get_dist(gt, output):
    dist_metric = np.dot(gt, output)
    dist_metric_norm = dist_metric/len(gt)

    return dist_metric_norm


def s_data_run():
    dist_list = []
    precision_list = []
    result_params_list = []
    em_t = []
    g_t = []
    gf_t = []
    ground_truth = [0, 1]*(obj_number/2)
    for pi in pi_list:

        print 'pi: {}'.format(pi)
        print '*****'

        for round in range(5):
            print round

            # ground_truth = [random.randint(0, len(possible_values)-1) for i in range(obj_number)]
            data, g_data = generator(cov_list, p_list, ground_truth, cl_size, pi, possible_values)

            mv, mv_pr = m_voting(data=data, truth_obj_list=ground_truth)
            print 'mv: {}'.format(mv)
            # print 'mv_pr: {}'.format(mv_pr)

            # t_em = time.time()
            em_d, em_it, accuracy_em, em_pr = em(data=data, truth_obj_list=ground_truth)
            print 'em: {}'.format(em_d)
            # print 'em_pr: {}'.format(em_pr)
            # ex_t_em = time.time() - t_em
            # em_t.append(ex_t_em)
            # print  em_it
            # print("--- %s seconds em ---" % (ex_t_em))

            # t_g = time.time()
            g_d, g_it, accuracy_g, g_pr = gibbs(data=data, truth_obj_list=ground_truth)
            print 'g: {}'.format(g_d)
            # print 'g_pr: {}'.format(g_pr)
            # ex_t_g = time.time() - t_g
            # g_t.append(ex_t_g)
            # print("--- %s seconds g ---" % (ex_t_g))

            # t_gf = time.time()
            gf_d, gf_it, accuracy_gf, pi_gf = gibbs_fuzzy(data=data, truth_obj_list=ground_truth)
            # ex_t_gf = time.time() - t_gf
            # gf_t.append(ex_t_gf)
            # print("--- %s seconds gf ---" % (ex_t_gf))
            print 'gf: {}'.format(gf_d)

            # em_f, em_f_it, accuracy_em_f, pi_em_f = em_fuzzy(data=data, truth_obj_list=ground_truth)
            # print 'em_f: {}'.format(em_f)
            # print '---'

            # dist_list.append([pi, mv, em_d, g_d, gf_d])
            # precision_list.append([pi, mv_pr, em_pr, g_pr])
            # pi_true_vector = [pi]*len(pi_em_f)
            # result_params_list.append([pi, get_dist(pi_true_vector, pi_em_f), get_dist(pi_true_vector, pi_gf),
            #                            get_dist(accuracy_list, [0.5]*s_number), get_dist(accuracy_list, accuracy_em),
            #                            get_dist(accuracy_list, accuracy_em_f), get_dist(accuracy_list, accuracy_gf)])

    # df_param = pd.DataFrame(data=result_params_list, columns=['pi', 'pi_em_f', 'pi_gf', 'ac_mv', 'ac_em', 'ac_em_f', 'ac_gf'])
    # df_dist = pd.DataFrame(data=dist_list, columns=['pi', 'mv', 'em', 'g', 'gf'])
    # df_precision = pd.DataFrame(data=precision_list, columns=['pi', 'mv_pr', 'em_pr', 'g_pr'])# 'em_f', 'gf'])
    # df_dist.to_csv('outputs/dist_5v2_{}_{}.csv'.format(s_number, obj_number))
    # df_precision.to_csv('outputs/precision_{}_{}.csv'.format(s_number, obj_number))
    # df_param.to_csv('output_param.csv')


def flights_data_run():
    gt_data = pd.read_csv('../data/flight/cleaned_data/gt_b_data.csv', index_col=0)
    data = pd.read_csv('../data/flight/cleaned_data/b_data.csv', index_col=0)
    ground_truth_list = list(gt_data.V.values)
    data = data[data.O.isin(range(len(ground_truth_list)))]

    mv, mv_pr = m_voting(data=data, truth_obj_list=ground_truth_list)
    print 'mv: {}'.format(mv)
    print 'mv_pr: {}'.format(mv_pr)

    em_d, em_it, accuracy_em, em_pr = em(data=data, truth_obj_list=ground_truth_list)
    print 'em: {}'.format(em_d)
    print 'em_pr: {}'.format(em_pr)


if __name__ == '__main__':
    s_data_run()
    # flights_data_run()
