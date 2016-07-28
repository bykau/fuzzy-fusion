import numpy as np
from mv import majority_voting
from em import expectation_maximization
from mcmc import mcmc
from f_mcmc import f_mcmc
import time
import pandas as pd
from util import accu_G


work_dir = '/home/bykau/Dropbox/Fuzzy/'

n_runs = 10


def confuse(Psi, conf_prob, GT, custom_Nc = None):
    M = len(Psi)
    gt_obj = list(GT.keys())
    if custom_Nc is None:
        Nc = len(gt_obj)
    else:
        Nc = custom_Nc
    Cl = {}
    for i in range(Nc/2):
        Cl[gt_obj[i]] = {'id': i, 'other': gt_obj[Nc/2+i]}
        Cl[gt_obj[Nc/2+i]] = {'id': i, 'other': gt_obj[i]}
    c_Psi = [[] for obj in range(M)]
    GT_G = {}
    for obj in Cl.keys():
        GT_G[obj] = {}

    for obj in range(M):
        if obj in Cl:
            for s, val in Psi[obj]:
                # check that a confused source hasn't voted already on the 'other' object
                if np.random.rand() >= conf_prob and s not in [x[0] for x in Psi[Cl[obj]['other']]] and val == GT[obj]:
                    c_Psi[Cl[obj]['other']].append((s, val))
                    GT_G[Cl[obj]['other']][s] = 0
                else:
                    c_Psi[obj].append((s, val))
                    GT_G[obj][s] = 1
        else:
            for s, val in Psi[obj]:
                c_Psi[obj].append((s, val))

    return GT_G, Cl, c_Psi


def load_dataset():
    Psi = []
    M = 0
    Ns = []
    with open(work_dir + 'data.txt') as f:
        for line in f:
            obj_votes = []
            vals = line.strip().split('\t')
            N = len(vals)
            Ns.append(N)
            for s in range(N):
                if vals[s] != '':
                    obj_votes.append((s, 'O'+str(M)+'_'+vals[s]))
            Psi.append(obj_votes)
            M += 1

    # there is a varying number of sources per object (apparently, a data quality issue), so we choose the max number of
    # sources.
    N = max(Ns)

    GT = {}
    with open(work_dir + 'truth_sample.txt') as f:
        for line in f:
            vals = line.strip().split('\t')
            if len(vals) == 2:
                GT[int(vals[0])] = 'O'+vals[0]+'_'+vals[1]
    return N, M, Psi, GT


def properties():
    """
    Print the flight dataset properties.
    """
    N, M, Psi, GT = load_dataset()
    print('# of sources: {}'.format(N))
    print('# of object: {}'.format(M))
    obs_n = 0
    V = [0 for obj in range(M)]
    for obj in range(M):
        obs_n += len(Psi[obj])
        V[obj] = len(set([val for s, val in Psi[obj]]))
    print('# of observations: {}'.format(obs_n))
    print('average # of values per object: {:1.3f}'.format(np.average(V)))
    print('min # of values per object: {:1.3f}'.format(min(V)))
    print('max # of values per object: {:1.3f}'.format(max(V)))


def accuracy():
    """
    Vary the confusion probability on real data.
    """
    N, M, Psi, GT = load_dataset()

    # inject confusions
    conf_probs = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    mcmc_params = {'N_iter': 10, 'burnin': 1, 'thin': 2, 'FV': 3}
    res = {'mv':[], 'em': [], 'mcmc': [], 'f_mcmc':[], 'confusion probability': conf_probs,
           'mv_std': [], 'em_std': [], 'mcmc_std': [], 'f_mcmc_std': [], 'G': [], 'G_std': []}
    for conf_prob in conf_probs:
        runs = [[], [], [], [], []]
        for run in range(n_runs):
            GT_G, Cl, cPsi = confuse(Psi, 1-conf_prob, GT)

            start = time.time()
            mv_p = majority_voting(cPsi)
            mv_t = time.time() - start

            start = time.time()
            em_A, em_p = expectation_maximization(N, M, cPsi)
            em_t = time.time() - start

            start = time.time()
            mcmc_A, mcmc_p = mcmc(N, M, cPsi, mcmc_params)
            mcmc_t = time.time() - start

            start = time.time()
            f_mcmc_A, f_mcmc_p, f_mcmc_G = f_mcmc(N, M, cPsi, Cl, mcmc_params)
            f_mcmc_t = time.time() - start

            mv_accu = np.average([mv_p[obj][GT[obj]] for obj in GT.keys()])
            em_accu = np.average([em_p[obj][GT[obj]] for obj in GT.keys()])
            mcmc_accu = np.average([mcmc_p[obj][GT[obj]] for obj in GT.keys()])
            f_mcmc_accu = np.average([f_mcmc_p[obj][GT[obj]] for obj in GT.keys()])

            runs[0].append(mv_accu)
            runs[1].append(em_accu)
            runs[2].append(mcmc_accu)
            runs[3].append(f_mcmc_accu)
            runs[4].append(accu_G(f_mcmc_G, GT_G))

        res['mv'].append(np.average(runs[0]))
        res['em'].append(np.average(runs[1]))
        res['mcmc'].append(np.average(runs[2]))
        res['f_mcmc'].append(np.average(runs[3]))
        res['G'].append(np.average(runs[4]))

        res['mv_std'].append(np.std(runs[0]))
        res['em_std'].append(np.std(runs[1]))
        res['mcmc_std'].append(np.std(runs[2]))
        res['f_mcmc_std'].append(np.std(runs[3]))
        res['G_std'].append(np.average(runs[4]))

        print('{}\tmv: {:1.4f}\tem: {:1.4f}\tmcmc: {:1.4f}\tf_mcmc: {:1.4f}'.format(conf_prob,
                                                                                                      mv_accu,
                                                                                                      em_accu,
                                                                                                      mcmc_accu,
                                                                                                      f_mcmc_accu
                                                                                                      )
              )

    pd.DataFrame(res).to_csv(work_dir + 'flight_accuracy.csv', index=False)


def efficiency():
    """
    Efficiency as the number of clusters growing.
    """
    N, M, Psi, GT = load_dataset()

    # inject confusions
    Ncs = [10, 100, 1000, 10000]
    mcmc_params = {'N_iter': 10, 'burnin': 1, 'thin': 2, 'FV': 3}
    res = {'mv': [],
           'mv std': [],
           'em': [],
           'em std': [],
           'mcmc': [],
           'mcmc std': [],
           'f_mcmc': [],
           'f_mcmc std': [],
           'number of objects with confusions': Ncs}
    for Nc in Ncs:
        times = [[], [], [], []]
        for run in range(n_runs):
            GT_G, Cl, cPsi = confuse(Psi, 0.8, GT, Nc)

            start = time.time()
            mv_p = majority_voting(cPsi)
            times[0].append(time.time() - start)

            start = time.time()
            em_A, em_p = expectation_maximization(N, M, cPsi)
            times[1].append(time.time() - start)

            start = time.time()
            mcmc_A, mcmc_p = mcmc(N, M, cPsi, mcmc_params)
            times[2].append(time.time() - start)

            start = time.time()
            f_mcmc_A, f_mcmc_p, f_mcmc_G = f_mcmc(N, M, cPsi, Cl, mcmc_params)
            times[3].append(time.time() - start)

        res['mv'].append(np.average(times[0]))
        res['em'].append(np.average(times[1]))
        res['mcmc'].append(np.average(times[2]))
        res['f_mcmc'].append(np.average(times[3]))

        res['mv std'].append(np.std(times[0]))
        res['em std'].append(np.std(times[1]))
        res['mcmc std'].append(np.std(times[2]))
        res['f_mcmc std'].append(np.std(times[3]))

        print('{}\tmv: {:1.4f}\tem: {:1.4f}\tmcmc: {:1.4f}\tf_mcmc: {:1.4f}'.format(Nc,
                                                                                                      np.average(times[0]),
                                                                                                      np.average(times[1]),
                                                                                                      np.average(times[2]),
                                                                                                      np.average(times[3])
                                                                                                      )
              )

    pd.DataFrame(res).to_csv(work_dir + 'flight_efficiency.csv', index=False)


if __name__ == '__main__':
    #accuracy()
    efficiency()
    #properties()



