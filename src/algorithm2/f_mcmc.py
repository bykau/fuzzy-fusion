import numpy as np
import math
from numpy.random import beta
from util import fPsi, log_likelihood
from collections import defaultdict
from generator import synthesize
from em import expectation_maximization


work_dir = '/Users/bykau/Desktop/Fuzzy/'


def f_mcmc(N, M, Psi, Cl, params):
    """
    MCMC inference with fuzzy observations.
    :param N:
    :param M:
    :param Psi: observations
    :param Cl: clusters where observations might be confused
    :return:
    """
    K = len(Cl)
    N_iter = params['N_iter']
    burnin = params['burnin']
    thin = params['thin']
    FV = params['FV']

    # init accuracies
    A = np.random.uniform(0.8, 1, N)

    # init confusions, for now we start with no confusions
    G = {}
    for obj in Cl.keys():
        G[obj] = {}
        for s, val in Psi[obj]:
            G[obj][s] = 1

    # init cluster Pis (confusion probs)
    Pi = np.random.uniform(0.75, 0.85, K)
    f_Psi, f_inv_Psi = fPsi(N, M, Psi, G, Cl)

    # MCMC sampling
    sample_size = 0.0
    f_mcmc_p = [defaultdict(float) for x in range(M)]
    # output G
    f_mcmc_G = {}
    for obj in Cl.keys():
        f_mcmc_G[obj] = {}
        for s, val in Psi[obj]:
            f_mcmc_G[obj][s] = [0.0, 0.0]

    for _iter in range(N_iter):
        # update objects
        p = []
        for obj in range(M):
            # a pass to detect all values of an object
            C = {}
            for s, val in f_Psi[obj]:
                C[val] = 0.0
            # total number of values
            V = len(C)

            # a pass to compute value confidences
            for s, val in f_Psi[obj]:
                for v in C.keys():
                    if v == val:
                        C[v] += math.log(A[s])
                    else:
                        C[v] += math.log((1-A[s])/(V-1))

            # compute probs
            # normalize
            norm = 0.0
            for val in C.keys():
                norm += math.exp(C[val])
            for val in C.keys():
                C[val] = math.exp(C[val])/norm
            p.append(C)

        # draw object values
        O = []
        for x in p:
            if len(x) > 0:
                vals = []
                probs = []
                for val, prob in x.iteritems():
                    vals.append(val)
                    probs.append(prob)

                O.append(vals[np.where(np.random.multinomial(1, probs) == 1)[0][0]])
            else:
                # if there are now values per object
                O.append(None)


        # update sources
        for source_id in range(N):
            beta_0 = 0
            beta_1 = 0
            for obj, val in f_inv_Psi[source_id]:
                if val == O[obj]:
                    beta_0 += 1
                else:
                    beta_1 += 1
            A[source_id] = beta(beta_0 + 4, beta_1 + 1)

        # update confusions
        for obj in Cl.keys():
            k = Cl[obj]['id']

            # make a pass to detect the total number of objects
            uniq_vals = set()
            for s, val in Psi[obj]:
                uniq_vals.add(val)
            for s, val in Psi[Cl[obj]['other']]:
                uniq_vals.add(val)
            V = len(uniq_vals)

            for s, val in Psi[obj]:
                pG = [0.0, 0.0]
                # if G[obj][s] == 1:
                if val == O[obj]:
                    pG[1] = Pi[k]*A[s]
                else:
                    pG[1] = Pi[k]*(1-A[s])/(V-1+FV)
                if val == O[Cl[obj]['other']]:
                    pG[0] = (1-Pi[k])*A[s]
                else:
                    pG[0] = (1-Pi[k])*(1-A[s])/(V-1+FV)
                norm = pG[0]+pG[1]
                pG[1] /= norm
                pG[0] /= norm
                # draw from Bernoulli
                G[obj][s] = int(np.random.rand() < pG[1])

        # update cluster counts
        nPi = [[0, 0] for k in range(K)]
        for obj in G.keys():
            if obj in Cl:
                k = Cl[obj]['id']
                for s in G[obj].keys():
                    if G[obj][s] == 1:
                        nPi[k][0] += 1
                    else:
                        nPi[k][1] += 1
        # draw from Beta distribution
        for k in range(K):
            Pi[k] = beta(nPi[k][0] + 8, nPi[k][1] + 2)

        # update observation matrix
        f_Psi, f_inv_Psi = fPsi(N, M, Psi, G, Cl)

        if _iter > burnin and _iter % thin == 0:
            sample_size += 1
            for obj in range(M):
                f_mcmc_p[obj][O[obj]] += 1
            for obj in G.keys():
                for s in G[obj].keys():
                    if G[obj][s] == 1:
                        f_mcmc_G[obj][s][1] += 1
                    else:
                        f_mcmc_G[obj][s][0] += 1
    # mcmc output
    for p in f_mcmc_p:
        for val in p.keys():
            p[val] /= sample_size
    f_mcmc_A = [0.0 for s in range(N)]
    for s in range(N):
        for obj, val in f_inv_Psi[s]:
            # TODO take advantage of priors (as in Zhao et al., 2012)
            f_mcmc_A[s] += f_mcmc_p[obj][val]
        f_mcmc_A[s] /= (0.0+len(f_inv_Psi[s]))
    for obj in f_mcmc_G.keys():
        for s in f_mcmc_G[obj].keys():
            total = f_mcmc_G[obj][s][0] + f_mcmc_G[obj][s][1]
            f_mcmc_G[obj][s][0] /= total
            f_mcmc_G[obj][s][1] /= total

    return f_mcmc_A, f_mcmc_p, f_mcmc_G