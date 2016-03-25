import numpy as np
import random
import copy

max_rounds = 30
eps = 0.001


def init_var(data, accuracy):
    observ_val = []
    s_number = len(accuracy.S)
    accuracy_list = list(accuracy.A)
    accuracy_ind = sorted(data.S.drop_duplicates())
    obj_index_list = sorted(data.O.drop_duplicates())
    for obj_index in obj_index_list:
        observ_val.append(sorted(list(set(data[data.O == obj_index].V))))
    random.shuffle(obj_index_list)
    random.shuffle(accuracy_ind)
    var_index = [obj_index_list, accuracy_ind]
    return [observ_val, var_index, accuracy_list, s_number]


def get_init_prob(data):
    init_prob = []
    obj_index_list = sorted(data.O.drop_duplicates())
    for obj_index in obj_index_list:
        init_prob.append([0.5, 0.5])
    return init_prob


#for randomize initial prob
def run_float(scalar, vector_size):
    random_vector = [random.random() for i in range(vector_size)]
    random_vector_sum = sum(random_vector)
    random_vector = [scalar*i/random_vector_sum for i in random_vector]
    return random_vector


def get_factor(prob, psi, G, accuracy_list, obj_index, v):
    factor = 0.
    possible_values = [0, 1]
    for g in G.iterrows():
        g = g[1]
        if g.Oj == obj_index and g.Oi == obj_index and psi.O == obj_index:
            if psi.V == v:
                factor += accuracy_list[psi.S]*g.P*prob[obj_index][v]
            else:
                factor += (1-accuracy_list[psi.S])*g.P*prob[obj_index][v]
        elif g.Oj != obj_index and g.Oi == obj_index and psi.O == obj_index:
            for v_true in possible_values:
                if psi.V == v_true:
                    factor += accuracy_list[psi.S]*g.P*prob[obj_index][v]*prob[int(g.Oj)][v_true]
                else:
                    factor += (1-accuracy_list[psi.S])*g.P*prob[obj_index][v]*prob[int(g.Oj)][v_true]
        elif g.Oi != obj_index:
            if g.Oj == obj_index:
                if psi.V == v:
                    factor += accuracy_list[psi.S]*g.P*prob[obj_index][v]
                else:
                    factor += (1-accuracy_list[psi.S])*g.P*prob[obj_index][v]
            else:
                for v_true in possible_values:
                    if psi.V == v_true:
                        factor += accuracy_list[psi.S]*g.P*prob[obj_index][v]*prob[int(g.Oj)][v_true]
                    else:
                        factor += (1-accuracy_list[psi.S])*g.P*prob[obj_index][v]*prob[int(g.Oj)][v_true]
    return factor


def get_prob(prob, data, g_data, accuracy_list, obj_index):
    G = g_data[g_data.Oi.isin(g_data.Oi[g_data.Oj == obj_index])]
    Psi = data[data.O.isin(G.Oi.drop_duplicates())]
    possible_values = [0, 1]
    a, b = 1., 1.
    for psi in Psi.iterrows():
        psi = psi[1]
        g_obj = G[G.Oi == psi.O]
        a *= get_factor(prob=prob, psi=psi, G=g_obj,
                        accuracy_list=accuracy_list, obj_index=obj_index, v=possible_values[0])
        b *= get_factor(prob=prob, psi=psi, G=g_obj,
                        accuracy_list=accuracy_list, obj_index=obj_index, v=possible_values[1])

    prob = [a/(a+b), b/(a+b)]
    return prob


def get_accuracy(data, g_data, prob, s_index, accuracy_old):
    Psi_s = data[data.S == s_index]
    G = g_data[g_data.Oi.isin(list(Psi_s.O))]
    size = len(Psi_s)
    p_sum = 0.
    claster_flag = True
    for psi in Psi_s.iterrows():
        psi = psi[1]
        a, b = 0., 0.
        for g in G.iterrows():
            g = g[1]
            if g.Oi != psi.O:
                continue
            if g.P == 1:
                psi_prob = prob[psi.O][psi.V]
                claster_flag = False
            else:
                for acc_val in [0, 1]:
                    if acc_val == psi.V:
                        a += prob[psi.O][psi.V]*accuracy_old*g.P
                    else:
                        b += (1-prob[psi.O][psi.V])*(1-accuracy_old)*g.P
        if claster_flag:
            psi_prob = a/(a+b)
        p_sum += psi_prob
    accuracy = p_sum/size
    return accuracy


def get_dist_metric(data, truth_obj_list, prob):
    possible_values = [0, 1]
    prob_gt = []
    val = []
    for obj_index in range(len(data.O.drop_duplicates())):
        val.append(possible_values)
        prob_gt.append([0]*2)
    for obj_ind, v_true in enumerate(truth_obj_list):
        for v_ind, v in enumerate(val[obj_ind]):
            if v == v_true:
                prob_gt[obj_ind][v_ind] = 1
    prob_gt_vector = []
    prob_vector = []
    for i in range(len(prob_gt)):
        prob_gt_vector += prob_gt[i]
        prob_vector += prob[i]
    dist_metric = np.dot(prob_gt_vector, prob_vector)
    dist_metric_norm = dist_metric/len(prob_gt)
    return dist_metric_norm


def gibbs_fuzzy(data, accuracy_data, g_data, truth_obj_list):
    dist_list = []
    iter_number_list = []
    for t in range(10):
        observ_val, var_index, accuracy_list, s_number = init_var(data=data, accuracy=accuracy_data)
        prob = get_init_prob(data=data)
        accuracy_delta = 0.3
        iter_number = 0
        while accuracy_delta > eps and iter_number < max_rounds:
            indexes = copy.deepcopy(var_index)
            accuracy_prev = copy.copy(accuracy_list)
            round_compl = False
            while not round_compl:
                        if len(indexes[0])!= 0 and len(indexes[1])!= 0:
                            r = random.randint(0, 1)
                            if r == 1:
                                o_ind = indexes[0].pop()
                                prob[o_ind] = get_prob(prob=prob, data=data, g_data=g_data,
                                                       accuracy_list=accuracy_list, obj_index=o_ind)
                            else:
                                s_index = indexes[1].pop()
                                accuracy_list[s_index] = get_accuracy(data=data, g_data=g_data, prob=prob,
                                                                      s_index=s_index, accuracy_old=accuracy_list[s_index])
                        elif len(indexes[0]) == 0 and len(indexes[1]) != 0:
                                s_index = indexes[1].pop()
                                accuracy_list[s_index] = get_accuracy(data=data, g_data=g_data, prob=prob,
                                                                      s_index=s_index, accuracy_old=accuracy_list[s_index])
                        elif len(indexes[0]) != 0 and len(indexes[1]) == 0:
                            o_ind = indexes[0].pop()
                            prob[o_ind] = get_prob(prob=prob, data=data, g_data=g_data,
                                                   accuracy_list=accuracy_list, obj_index=o_ind)
                        else:
                            round_compl = True
            iter_number += 1
            accuracy_delta = max([abs(k-l) for k, l in zip(accuracy_prev, accuracy_list)])
        dist_metric = get_dist_metric(data=data, truth_obj_list=truth_obj_list, prob=prob)
        dist_list.append(dist_metric)
        iter_number_list.append(iter_number)
    return [np.mean(dist_list), np.mean(iter_number_list)]
