import random
import sys
import time

import numpy as np
import pandas as pd

# sys.path.append('/home/evgeny/fuzzy-fusion/src/')
# sys.path.append('/home/evgeny/fuzzy-fusion/experiment/')
sys.path.append('/Users/Evgeny/wonderful_programming/fuzzy-fusion-venv/fuzzy-fusion/src/')
sys.path.append('/Users/Evgeny/wonderful_programming/fuzzy-fusion-venv/fuzzy-fusion/experiment/')
from generator.generator import generator
from algorithm.gibbs import gibbs
from algorithm.gibbs_fuzzy import gibbs_fuzzy
from algorithm.em import em
from algorithm.sums import sums
from algorithm.average_log import average_log
from algorithm.investment import investment
from algorithm.pooled_investment import pooled_investment
from algorithm.m_voting import m_voting
from algorithm.common import get_data

# number of sources
s_number = 10
# number of objects
obj_number = 1000
# cluster size
cl_size = 2
# possible values for an objects(for the data generator)
possible_values = range(5)
# sources coverage
cov_val_list = [0.7]#[0.7, 0.8, 0.9, 1.0]
# sources accuracy
p_val_list = [.7, .75, .8, .85, .9, .95, 1.]
# probability that sources don't confuse an object
pi = 1.


# experiment on synthetic data
def s_data_run():
    # list of algorithms accuracy
    alg_ac_list = []

    # run algorithms with different params(cov, accuracy of sources)
    # for the data generator
    for p in p_val_list:
        p_list = [p]*s_number
        for cov in cov_val_list:
            cov_list = [cov]*s_number
            print 's accuracy: {}'.format(p)
            print 'cov: {}'.format(cov)

            for round in range(10):
                print 'Round: {}'.format(round)
                ground_truth = dict([(i, random.randint(0, len(possible_values)-1)) for i in range(obj_number)])
                # currently the data generator output is pandas dataFrame
                data2, g_data = generator(cov_list, p_list, ground_truth, cl_size, pi, possible_values)
                # transform pandas dataFrame into dict format
                data = get_data(data=data2)

                # PRINT OUT ALGORITHMS ACCURACIES
                mv_ac = m_voting(data=data, gt=ground_truth)
                print 'MV_ac: {}'.format(mv_ac)

                sum_ac = sums(data=data, gt=ground_truth, s_number=s_number)
                print 'SUM_ac: {}'.format(sum_ac)

                al_ac = average_log(data=data, gt=ground_truth, s_number=s_number)
                print 'AL_ac: {}'.format(al_ac)

                inv_ac = investment(data=data, gt=ground_truth, s_number=s_number)
                print 'INV_ac: {}'.format(inv_ac)
                #
                pInv_ac = pooled_investment(data=data, gt=ground_truth, s_number=s_number)
                print 'PINV_ac: {}'.format(pInv_ac)

                em_ac = em(data=data, gt=ground_truth,
                           accuracy_truth=p_list, s_number=s_number)
                print 'EM_ac: {}'.format(em_ac)

                g_ac = gibbs(data=data, gt=ground_truth,
                             accuracy_truth=p_list, s_number=s_number)
                print 'GB_ac: {}'.format(g_ac)

                # gf_ac = gibbs_fuzzy(data=data, gt=ground_truth,
                #                     accuracy_truth=p_list, s_number=s_number)
                # print 'FG_ac: {}'.format(gf_ac)

                alg_ac_list.append([p, mv_ac, sum_ac, al_ac, inv_ac, pInv_ac, em_ac, g_ac])
                print '---'

                # PRINT OUT ALGORITHMS DIST AND OTHER METRICS
                # mv, mv_pr = m_voting(data=data, gt=ground_truth)
                # print 'mv: {}'.format(mv)
                # print 'mv_pr: {}'.format(mv_pr)

                # t_em = time.time()
                # em_d, em_it, em_pr, em_ac_err = em(data=data, gt=ground_truth,
                #                                    accuracy_truth=p_list, s_number=s_number)
                # print 'em: {}'.format(em_d)
                # print 'em ac err: {}'.format(em_ac_err)
                # print 'em_pr: {}'.format(em_pr)
                # ex_t_em = time.time() - t_em
                # em_t.append(ex_t_em)
                # print  em_it
                # print("--- %s seconds em ---" % (ex_t_em))

                # t_g = time.time()
                # g_d, g_it, g_pr, g_ac_err = gibbs(data=data, gt=ground_truth,
                #                                   accuracy_truth=p_list, s_number=s_number)
                # print 'g: {}'.format(g_d)
                # print 'g ac err: {}'.format(g_ac_err)
                # print 'g_pr: {}'.format(g_pr)
                # ex_t_g = time.time() - t_g
                # g_t.append(ex_t_g)
                # print("--- %s seconds g ---" % (ex_t_g))
                # gf_d, gf_pr, gf_ac, gf_pi = gibbs_fuzzy(data=data, gt=ground_truth,
                #                                         accuracy_truth=p_list, s_number=s_number)
                # print 'gf: {}'.format(gf_d)
                # print 'gf_pr: {}'.format(gf_pr)
                #
                # print '---'
                # dist_list.append([p, cov, mv, em_d, g_d])
                # acc_err_list.append([p, cov, em_ac_err, g_ac_err])
    df_ac = pd.DataFrame(data=alg_ac_list, columns=['p', 'mv_ac', 'sums_ac', 'al_ac', 'inv_ac', 'pInv_ac', 'em_ac', 'g_ac'])
    # df_ac.to_csv('outputs/alg_ac_v5_{}_{}.csv'.format(s_number, obj_number), index=False)
    # df_dist = pd.DataFrame(data=dist_list, columns=['acc', 'cov', 'mv', 'em', 'g'])
    # df_dist.to_csv('outputs/dist_v5_{}_{}.csv'.format(s_number, obj_number), index=False)
    # df_acc_err = pd.DataFrame(data=acc_err_list, columns=['acc', 'cov', 'em_ac_err', 'g_ac_err'])
    # df_acc_err.to_csv('outputs/acc_v5_{}_{}.csv'.format(s_number, obj_number), index=False)


def flights_data_run():
    # import pandas as pd
    # ground_truth = pd.read_csv('../data/flight/data/gt.csv', low_memory=False)
    # data = pd.read_csv('../data/flight/data/data.csv', low_memory=False)
    # data_py = {}
    # objs_ind = sorted(data.O.drop_duplicates().values)
    # for obj in objs_ind:
    #     obj_s = list(data[data.O == obj].S.values)
    #     obj_vals = list(data[data.O == obj].V.values)
    #     data_py.update({obj: [obj_s, obj_vals]})
    # f = open('flights_data.py', 'w')
    # f.write(str(data_py))
    # f.close()

    from flights_data import flights
    from flights_gt import ground_truth

    s_number = 38

    mv, mv_pr = m_voting(data=flights, gt=ground_truth)
    print 'mv: {}'.format(mv)
    print 'mv_pr: {}'.format(mv_pr)

    t_em = time.time()
    em_d, em_it, em_pr, accuracy_em,  = em(data=flights, gt=ground_truth, s_number=s_number)
    print 'em: {}'.format(em_d)
    print 'em_pr: {}'.format(em_pr)
    ex_t_em = time.time() - t_em
    print("--- %s seconds em ---" % (ex_t_em))

    t_g = time.time()
    g_d, g_it, g_pr, g_ac_err = gibbs(data=flights, gt=ground_truth, s_number=s_number)
    print 'g: {}'.format(g_d)
    print 'g_pr: {}'.format(g_pr)
    ex_t_g = time.time() - t_g
    print("--- %s seconds g ---" % (ex_t_g))


def pop_data_run():
    # ground_truth = pd.read_csv('../data/population/data/pop_gt.csv')
    # gt_dict = dict(zip(ground_truth['O'].values, ground_truth['V']))
    # data_pd = pd.read_csv('../data/population/data/pop_data.csv', low_memory=False)
    # data_py = get_data(data=data_pd)
    # f = open('pop_data.py', 'w')
    # f.write(str(data_py))
    # f.close()
    # f2 = open('pop_gt.py', 'w')
    # f2.write(str(gt_dict))
    # f2.close()

    from pop_data import data
    from pop_gt import ground_truth

    s_number = 4216

    mv_ac = m_voting(data=data, gt=ground_truth)
    print 'MV_ac: {}'.format(mv_ac)

    inv_ac = investment(data=data, gt=ground_truth, s_number=s_number)
    print 'INV_ac: {}'.format(inv_ac)

    pInv_ac = pooled_investment(data=data, gt=ground_truth, s_number=s_number)
    print 'PINV_ac: {}'.format(pInv_ac)

    g_ac = gibbs(data=data, gt=ground_truth, s_number=s_number)
    print 'GB_ac: {}'.format(g_ac)

    em_ac = em(data=data, gt=ground_truth, s_number=s_number)
    print 'EM_ac: {}'.format(em_ac)


if __name__ == '__main__':
    s_data_run()
    # flights_data_run()
    # pop_data_run()
