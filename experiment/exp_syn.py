import random
import sys
import pandas as pd

sys.path.append('/Users/Evgeny/wonderful_programming/fuzzy-fusion-venv/fuzzy-fusion/src/')
sys.path.append('/Users/Evgeny/wonderful_programming/fuzzy-fusion-venv/fuzzy-fusion/experiment/')
from generator.generator import generator
from algorithm.gibbs import gibbs
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

                alg_ac_list.append([p, mv_ac, sum_ac, al_ac, inv_ac, pInv_ac, em_ac, g_ac])
                print '---'

    # create pandas dataFrame with algorithms outputs
    df_ac = pd.DataFrame(data=alg_ac_list, columns=['p', 'mv_ac', 'sums_ac', 'al_ac', 'inv_ac', 'pInv_ac', 'em_ac', 'g_ac'])
    # output to csv file
    df_ac.to_csv('outputs/alg_ac_v5_{}_{}.csv'.format(s_number, obj_number), index=False)


if __name__ == '__main__':
    s_data_run()
