import math


def log_likelihood(GT, M, Psi, A, p):
    """
    Computes the log likelihood of the Psi using A and p.
    """
    res = 0
    for obj_id in range(M):
        for source_id, value_id in Psi[obj_id]:
            if p[obj_id][value_id] == 1.0:
                p[obj_id][value_id] = 0.9999999
            if p[obj_id][value_id] == 0.0:
                p[obj_id][value_id] = 0.0000001
            if value_id == GT[obj_id]:
                res += math.log(A[source_id] * p[obj_id][value_id])
            else:
                res += math.log((1 - A[source_id]) * (1 - p[obj_id][value_id]))
    return res


def invert(N, M, Psi):
    """
    Inverts the observation matrix. Need for performance reasons.
    :param N:
    :param M:
    :param Psi:
    :return:
    """
    inv_Psi = [[] for s in range(N)]
    for obj in range(M):
        for s, val in Psi[obj]:
            inv_Psi[s].append((obj, val))
    return inv_Psi


def fPsi(N, M, Psi, G, Cl):
    """
    Computes the observation matrix based on known confusions.
    :param N:
    :param M:
    :param Psi:
    :param G: confusions
    :param Cl: clusters
    :return:
    """
    f_Psi = [[] for x in range(M)]
    for obj in range(M):
        for s, val in Psi[obj]:
            if obj in Cl:
                if G[obj][s] == 1:
                    f_Psi[obj].append((s, val))
                else:
                    f_Psi[Cl[obj]['other']].append((s, val))
            else:
                f_Psi[obj].append((s, val))
    return f_Psi, invert(N, M, f_Psi)


def accu_G(f_mcmc_G, GT_G):
    tp = 0.0
    total = 0.0
    for obj in GT_G.keys():
        for s in GT_G[obj]:
            tp += f_mcmc_G[obj][s][GT_G[obj][s]]
            total += 1

    return tp/total