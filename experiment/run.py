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
obj_number = 1000
cl_size = 2
possible_values = range(5)
cov_val_list = [0.7, 0.8, 0.9, 1.0]
p_val_list = [.7, .75, .8, .85, .9, .95, 1.]
pi = 1.


def get_dist(gt, output):
    dist_metric = np.dot(gt, output)
    dist_metric_norm = dist_metric/len(gt)

    return dist_metric_norm


def s_data_run():
    dist_list = []
    acc_err_list = []

    for p in p_val_list:
        p_list = [p]*s_number
        for cov in cov_val_list:
            cov_list = [cov]*s_number
            print 'accuracy: {}'.format(p)
            print 'cov: {}'.format(cov)

            for round in range(5):
                print round

                ground_truth = [random.randint(0, len(possible_values)-1) for i in range(obj_number)]
                data, g_data = generator(cov_list, p_list, ground_truth, cl_size, pi, possible_values)

                mv, mv_pr = m_voting(data=data, truth_obj_list=ground_truth)
                print 'mv: {}'.format(mv)
                # print 'mv_pr: {}'.format(mv_pr)

                # t_em = time.time()
                em_d, em_it, em_pr, em_ac_err = em(data=data, truth_obj_list=ground_truth, accuracy_truth=p_list)
                print 'em: {}'.format(em_d)
                print 'em ac err: {}'.format(em_ac_err)
                # print 'em_pr: {}'.format(em_pr)
                # ex_t_em = time.time() - t_em
                # em_t.append(ex_t_em)
                # print  em_it
                # print("--- %s seconds em ---" % (ex_t_em))

                # t_g = time.time()
                g_d, g_it, g_pr, g_ac_err = gibbs(data=data, truth_obj_list=ground_truth, accuracy_truth=p_list)
                print 'g: {}'.format(g_d)
                print 'g ac err: {}'.format(g_ac_err)
                # print 'g_pr: {}'.format(g_pr)
                # ex_t_g = time.time() - t_g
                # g_t.append(ex_t_g)
                # print("--- %s seconds g ---" % (ex_t_g))

                print '---'

                dist_list.append([p, cov, mv, em_d, g_d])
                acc_err_list.append([p, cov, em_ac_err, g_ac_err])

    df_dist = pd.DataFrame(data=dist_list, columns=['acc', 'cov', 'mv', 'em', 'g'])
    df_dist.to_csv('outputs/dist_v5_{}_{}.csv'.format(s_number, obj_number), index=False)
    df_acc_err = pd.DataFrame(data=acc_err_list, columns=['acc', 'cov', 'em_ac_err', 'g_ac_err'])
    df_acc_err.to_csv('outputs/acc_v5_{}_{}.csv'.format(s_number, obj_number), index=False)


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
