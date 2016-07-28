import numpy as np
import math
from numpy.random import beta
from collections import defaultdict
from util import invert


def mcmc(N, M, Psi, params):
    """
    MCMC for log-likelihood maximum search.
    :param N:
    :param M:
    :param Psi:
    :param inv_Psi:
    :return:
    """
    N_iter = params['N_iter']
    burnin = params['burnin']
    thin = params['thin']

    inv_Psi = invert(N, M, Psi)

    # random init
    A = np.random.uniform(0.8, 1.0, N)

    # MCMC sampling
    sample_size = 0.0
    mcmc_p = [defaultdict(float) for x in range(M)]
    for _iter in range(N_iter):
        # update objects
        p = []
        for obj in range(M):

            # a pass to detect all values of an object
            C = {}
            for s, val in Psi[obj]:
                C[val] = 0.0
            # total number of values
            V = len(C)

            # a pass to compute value confidences
            for s, val in Psi[obj]:
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
            for obj, val in inv_Psi[source_id]:
                if val == O[obj]:
                    beta_0 += 1
                else:
                    beta_1 += 1
            A[source_id] = beta(beta_0 + 4, beta_1 + 1)

        if _iter > burnin and _iter % thin == 0:
            sample_size += 1
            for obj in range(M):
                mcmc_p[obj][O[obj]] += 1

    # mcmc output
    for p in mcmc_p:
        for val in p.keys():
            p[val] /= sample_size
    mcmc_A = [0.0 for s in range(N)]
    for s in range(N):
        for obj, val in inv_Psi[s]:
            # TODO take advantage of priors (as in Zhao et al., 2012)
            mcmc_A[s] += mcmc_p[obj][val]
        mcmc_A[s] /= (0.0+len(inv_Psi[s]))

    return mcmc_A, mcmc_p