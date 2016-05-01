import sys
import random
import time
import pandas as pd
import numpy as np
# sys.path.append('/home/evgeny/fuzzy-fusion/src/')
sys.path.append('/Users/Evgeny/wonderful_programming/fuzzy-fusion-venv/fuzzy-fusion/src/')
from generator.generator import generator
from algorithm.gibbs_fuzzy import gibbs_fuzzy
from algorithm.gibbs import gibbs
from algorithm.em import em
from algorithm.em_fuzzy import em_fuzzy
from algorithm.m_voting import m_voting


s_number = 5
obj_number = 20
cl_size = 2
possible_values = [0, 1]
cov_list = [0.7]*s_number
p_list = [0.8]*s_number


def get_dist(gt, output):
    dist_metric = np.dot(gt, output)
    dist_metric_norm = dist_metric/len(gt)

    return dist_metric_norm


if __name__ == '__main__':
    result_list = []
    result_params_list = []
    em_t = []
    g_t = []
    gf_t = []
    ground_truth = [0, 1]*(obj_number/2)
    for pi in [1., 0.95, 0.9, 0.85, 0.8, 0.75, 0.7]:# 0.65, 0.6, 0.55, 0.5]:

        print 'pi: {}'.format(pi)
        print '*****'

        for round in range(5):
            print round

            # ground_truth = [random.randint(0, len(possible_values)-1) for i in range(obj_number)]
            data, g_data = generator(cov_list, p_list, ground_truth, cl_size, pi, possible_values)

            m_v = m_voting(data=data, truth_obj_list=ground_truth)
            print 'm_v: {}'.format(m_v)

            # t_em = time.time()
            em_d, em_it, accuracy_em = em(data=data, truth_obj_list=ground_truth, values=possible_values)
            print 'em: {}'.format(em_d)

            # ex_t_em = time.time() - t_em
            # em_t.append(ex_t_em)
            # print("--- %s seconds em ---" % (ex_t_em))

            # t_g = time.time()
            g_d, g_it, accuracy_g = gibbs(data=data, truth_obj_list=ground_truth)
            print 'g: {}'.format(g_d)
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
            print '---'

            result_list.append([pi, m_v, em_d, g_d, gf_d])
            # pi_true_vector = [pi]*len(pi_em_f)
            # result_params_list.append([pi, get_dist(pi_true_vector, pi_em_f), get_dist(pi_true_vector, pi_gf),
            #                            get_dist(accuracy_list, [0.5]*s_number), get_dist(accuracy_list, accuracy_em),
            #                            get_dist(accuracy_list, accuracy_em_f), get_dist(accuracy_list, accuracy_gf)])

    # df_param = pd.DataFrame(data=result_params_list, columns=['pi', 'pi_em_f', 'pi_gf', 'ac_mv', 'ac_em', 'ac_em_f', 'ac_gf'])
    df_data = pd.DataFrame(data=result_list, columns=['pi', 'mv', 'em', 'g', 'g_f'])# 'em_f', 'gf'])
    df_data.to_csv('output_data.csv')
    # df_param.to_csv('output_param.csv')
