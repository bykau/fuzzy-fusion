import numpy as np
import time
from mv import majority_voting
from em import expectation_maximization
from mcmc import mcmc
from f_mcmc import f_mcmc
from generator import synthesize
import pandas as pd

work_dir = '/home/bykau/Dropbox/Fuzzy/'
n_runs = 10


def adapter_input(Psi):
    Psi_new = {}
    for obj_ind, obj_data in enumerate(Psi):
        obj_s, obj_v = [], []
        for s, v in obj_data:
            obj_s.append(s)
            obj_v.append(v)
        obj_data_new = {obj_ind: [obj_s, obj_v]}
        Psi_new.update(obj_data_new)
    return Psi_new


def adapter_output(belief, data):
    val_p = []
    for obj_ind in sorted(belief.keys()):
        possible_values = sorted(list(set(data[obj_ind][1])))
        obj_p = map(lambda x: 0.0 if x != 1. else x, belief[obj_ind])
        val_p.append(dict(zip(possible_values, obj_p)))
    return val_p


def accuracy():
    """
    Vary the confusion probability on synthetic data.
    """
    # number of sources
    N = 30
    # number of objects
    M = 5000
    # number of values per object
    V = 50
    # synthetically generated observations
    density = 0.3
    accuracy = 0.8

    mcmc_params = {'N_iter': 10, 'burnin': 1, 'thin': 2, 'FV': 0}
    conf_probs = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    res = {'mv': [], 'em': [], 'mcmc': [], 'f_mcmc': [], 'mv std': [], 'em std': [], 'mcmc std': [], 'f_mcmc std': [], 'confusion probability': conf_probs}
    for conf_prob in conf_probs:
        GT, GT_G, Cl, Psi = synthesize(N, M, V, density, accuracy, 1-conf_prob)

        mv_accu, em_accu, mcmc_accu, f_mcmc_accu = [], [], [], []
        for run in range(n_runs):
            start = time.time()
            mv_p = majority_voting(Psi)
            mv_t = time.time() - start

            start = time.time()
            em_A, em_p = expectation_maximization(N, M, Psi)
            em_t = time.time() - start

            start = time.time()
            mcmc_A, mcmc_p = mcmc(N, M, Psi, mcmc_params)
            mcmc_t = time.time() - start

            start = time.time()
            f_mcmc_A, f_mcmc_p, f_mcmc_G = f_mcmc(N, M, Psi, Cl, mcmc_params)
            f_mcmc_t = time.time() - start

            mv_accu.append(np.average([mv_p[obj][GT[obj]] for obj in GT.keys()]))
            em_accu.append(np.average([em_p[obj][GT[obj]] for obj in GT.keys()]))
            mcmc_accu.append(np.average([mcmc_p[obj][GT[obj]] for obj in GT.keys()]))
            f_mcmc_accu.append(np.average([f_mcmc_p[obj][GT[obj]] for obj in GT.keys()]))

        res['mv'].append(np.average(mv_accu))
        res['mv std'].append(np.std(mv_accu))
        res['em'].append(np.average(em_accu))
        res['em std'].append(np.std(em_accu))
        res['mcmc'].append(np.average(mcmc_accu))
        res['mcmc std'].append(np.std(mcmc_accu))
        res['f_mcmc'].append(np.average(f_mcmc_accu))
        res['f_mcmc std'].append(np.std(f_mcmc_accu))

        print('confusion probability: {}, mv: {:1.4f}, em: {:1.4f}, mcmc: {:1.4f}, f_mcmc: {:1.4f}'.format(conf_prob,
                                                                                       np.average(mv_accu),
                                                                                       np.average(em_accu),
                                                                                       np.average(mcmc_accu),
                                                                                       np.average(f_mcmc_accu)
                                                                                       )
            )

    pd.DataFrame(res).to_csv(work_dir + 'synthetic_accuracy.csv', index=False)


def convergence():
    """
    Convergence of MCMC.
    """
    # number of sources
    N = 30
    # number of objects
    M = 5000
    # number of values per object
    V = 30
    # synthetically generated observations
    density = 0.3
    accuracy = 0.8
    conf_prob = 0.8

    GT, GT_G, Cl, Psi = synthesize(N, M, V, density, accuracy, conf_prob)
    res = {'accuracy': [], 'error': [], 'number of iterations': [5, 10, 30, 50, 100]}
    for p in [(3, 0, 1), (5, 0, 1), (10, 1, 2), (30, 5, 3), (50, 7, 5), (100, 10, 7)]:
        params = {'N_iter': p[0], 'burnin': p[1], 'thin': p[2], 'FV': 0}
        runs = []
        for run in range(n_runs):
            f_mcmc_A, f_mcmc_p, f_mcmc_G = f_mcmc(N, M, Psi, Cl, params)
            f_mcmc_accu = np.average([f_mcmc_p[obj][GT[obj]] for obj in GT.keys()])
            runs.append(f_mcmc_accu)
        res['accuracy'].append(np.average(runs))
        res['error'].append(np.std(runs))

        print('p: {}, accu: {}, std: {}'.format(p, np.average(runs), np.std(runs)))

    pd.DataFrame(res).to_csv(work_dir + 'synthetic_convergence.csv', index=False)


def values():
    """
    Vary the number of distinct values V.
    """
    # number of sources
    N = 30
    # number of objects
    M = 5000
    # synthetically generated observations
    density = 0.3
    accuracy = 0.8
    conf_prob = 0.8
    Vs = [2, 4, 8, 16, 32, 64, 128]
    params = {'N_iter': 10, 'burnin': 1, 'thin': 2, 'FV': 0}
    res = {'accuracy': [], 'std': [], 'number of distinct values per object': Vs}
    for V in Vs:
        GT, GT_G, Cl, Psi = synthesize(N, M, V, density, accuracy, conf_prob)
        f_mcmc_accu = []
        for run in range(n_runs):
            f_mcmc_A, f_mcmc_p, f_mcmc_G = f_mcmc(N, M, Psi, Cl, params)
            f_mcmc_accu.append(np.average([f_mcmc_p[obj][GT[obj]] for obj in GT.keys()]))
        res['accuracy'].append(np.average(f_mcmc_accu))
        res['std'].append(np.std(f_mcmc_accu))
        print('V: {}, accu: {:1.4f}'.format(V, np.average(f_mcmc_accu)))

    pd.DataFrame(res).to_csv(work_dir + 'synthetic_values.csv', index=False)


if __name__ == '__main__':
    accuracy()
    #convergence()
    #values()

