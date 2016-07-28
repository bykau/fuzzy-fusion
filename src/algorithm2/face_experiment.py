import pandas as pd
import numpy as np
from em import expectation_maximization
from mv import majority_voting
from f_mcmc import f_mcmc
from mcmc import mcmc
import numpy as np
import random
from util import accu_G

work_dir = '/Users/bykau/Desktop/Fuzzy/'
n_runs = 10


def load_data():
    M = 60
    face1 = pd.read_csv(work_dir + 'Faces.Test 1.csv').replace('I don\'t know', np.nan)
    face2 = pd.read_csv(work_dir + 'Faces.Test 2.csv').replace('I don\'t know', np.nan)
    face3 = pd.read_csv(work_dir + 'Faces.Test 3.csv').replace('I don\'t know', np.nan)
    face1.columns = ['Timestamp', 'User'] + [str(x) for x in range(0, 60, 2)]
    face2.columns = ['Timestamp', 'User'] + [str(x) for x in range(1, 60, 2)]
    face3.columns = ['Timestamp', 'User'] + [2 * x if x % 2 == 0 else (2 * x + 1) for x in range(15)] + [
        (30 + 2 * x) if x % 2 == 0 else (31 + 2 * x) for x in range(15)]
    GT = {}
    for i in range(0, M, 2):
        GT[i] = face1.to_dict()[str(i)][0]
    for i in range(1, M, 2):
        GT[i] = face2.to_dict()[str(i)][0]
    face1 = face1[face1.User != 'Ground Truth']
    face2 = face2[face2.User != 'Ground Truth']
    face3 = face3[face3.User != 'Ground Truth']
    Psi = [[] for x in range(M)]
    offset = 0
    users = {}
    for f in [face1, face2, face3]:
        for obj in f.to_dict().keys():
            if obj not in ['User', 'Timestamp']:
                for s in f.to_dict()[obj]:
                    if (s + offset - 1) not in users:
                        users[s + offset - 1] = f.to_dict()['User'][s]
                    val = f.to_dict()[obj][s]
                    if val is not np.nan:
                        Psi[int(obj)].append((offset + s - 1, val))
        offset += f.shape[0]
    N = len(users)
    # Cl = {}
    # for x in range(0, M, 2):
    #     Cl[x] = {'other': (x + 1), 'id': x / 2}
    #     Cl[x + 1] = {'other': x, 'id': x / 2}
    Cl = {56: {'other': 57, 'id': 0},
          57: {'other': 56, 'id': 0},
          42: {'other': 43, 'id': 1},
          43: {'other': 42, 'id': 1},
          38: {'other': 39, 'id': 2},
          39: {'other': 38, 'id': 2},
          36: {'other': 37, 'id': 3},
          37: {'other': 36, 'id': 3},
          # 28: {'other': 29, 'id': 4},
          # 29: {'other': 28, 'id': 4},
          22: {'other': 23, 'id': 5},
          23: {'other': 22, 'id': 5}
          }

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
    n_conf = 0
    for obj in range(M):
        for s, val in Psi[obj]:
            if obj in Cl and val == GT[Cl[obj]['other']]:
                n_conf += 1
                # print(obj, users[s], GT[obj], val)
    print('# of confusions: {}'.format(n_conf))

    # # add two sources which always provide true values
    # for obj in range(0, M, 2):
    #     Psi[obj].append((N, GT[obj]))
    #     Psi[obj].append((N+1, GT[obj]))
    # for obj in range(1, M, 2):
    #     Psi[obj].append((N + 2, GT[obj]))
    #     Psi[obj].append((N+3, GT[obj]))
    # N += 4

    # compute a cleaned set of observations
    cleaned_Psi = [[] for obj in range(M)]
    for obj in range(M):
        for s, val in Psi[obj]:
            if obj in Cl and val == GT[Cl[obj]['other']]:
                cleaned_Psi[Cl[obj]['other']].append((s, val))
            else:
                cleaned_Psi[obj].append((s, val))

    for obj in range(M):
        print(GT[obj], obj, Psi[obj])
        if obj % 2 != 0:
            print
    return N, M, Psi, cleaned_Psi, Cl, GT


def accuracy():
    N, M, Psi, cleaned_Psi, Cl, GT = load_data()
    res = {'accuracy': [],
           'std': [],
           'methods': ['mv', 'em', 'mcmc', 'f_mcmc']}
    runs = [[], [], [], []]
    for run in range(n_runs):
        mv_p = majority_voting(Psi)
        em_A, em_p = expectation_maximization(N, M, Psi)
        mcmc_A, mcmc_p = mcmc(N, M, Psi, {'N_iter': 10, 'burnin': 1, 'thin': 2})
        f_mcmc_A, f_mcmc_p, f_mcmc_G = f_mcmc(N, M, Psi, Cl, {'N_iter': 30, 'burnin': 5, 'thin': 3, 'FV': 4})

        mv_hits = []
        em_hits = []
        mcmc_hits = []
        f_mcmc_hits = []
        for obj in range(M):
            if len(Psi[obj]) > 0:
                mv_hits.append(mv_p[obj][GT[obj]])
                em_hits.append(em_p[obj][GT[obj]])
                mcmc_hits.append(mcmc_p[obj][GT[obj]])
                f_mcmc_hits.append(f_mcmc_p[obj][GT[obj]])

        runs[0].append(np.average(mv_hits))
        runs[1].append(np.average(em_hits))
        runs[2].append(np.average(mcmc_hits))
        runs[3].append(np.average(f_mcmc_hits))

    print('mv: {:1.4f}+-{:1.4f}'.format(np.average(runs[0]), np.std(runs[0])))
    print('em: {:1.4f}+-{:1.4f}'.format(np.average(runs[1]), np.std(runs[1])))
    print('mcmc: {:1.4f}+-{:1.4f}'.format(np.average(runs[2]), np.std(runs[2])))
    print('f_mcmc: {:1.4f}+-{:1.4f}'.format(np.average(runs[3]), np.std(runs[3])))
    res['accuracy'].append(np.average(runs[0]))
    res['accuracy'].append(np.average(runs[1]))
    res['accuracy'].append(np.average(runs[2]))
    res['accuracy'].append(np.average(runs[3]))
    res['std'].append(np.std(runs[0]))
    res['std'].append(np.std(runs[1]))
    res['std'].append(np.std(runs[2]))
    res['std'].append(np.std(runs[3]))

    for obj in range(M):
        for s, val in Psi[obj]:
            if obj in Cl and val == GT[Cl[obj]['other']]:
                print(GT[obj], f_mcmc_G[obj][s])

    #pd.DataFrame(res).to_csv(work_dir + 'face_accuracy.csv', index=False)


if __name__ == '__main__':
    accuracy()
