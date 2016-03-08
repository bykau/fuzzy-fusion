import pandas as pd
import random


def init_var(data, accuracy):
    observ_val = []
    s_number = len(accuracy.S)
    accuracy_list = list(accuracy.A)
    accuracy_ind = sorted(data.S.drop_duplicates())
    obj_index_list = sorted(data.O.drop_duplicates())
    for obj_index in obj_index_list:
        possible_values = sorted(list(set(data[data.O == obj_index].V)))
        observ_val.append(possible_values)
    random.shuffle(obj_index_list)
    random.shuffle(accuracy_ind)
    var_index = [obj_index_list, accuracy_ind]
    return [observ_val, var_index, accuracy_list, s_number]


def get_init_prob(data):
    init_prob = []
    obj_index_list = sorted(data.O.drop_duplicates())
    for obj_index in obj_index_list:
        l = len(sorted(list(set(data[data.O == obj_index].V))))
        # init_prob.append(run_float(scalar=1, vector_size=l))
        # TO DO
        init_prob.append([1./2]*2)
    return init_prob


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
                factor += (1-accuracy_list[psi.S])*g.P*(1-prob[obj_index][v])
        elif g.Oj != obj_index and g.Oi == obj_index and psi.O == obj_index:
            for v_true in possible_values:
                if psi.V == v_true:
                    factor += accuracy_list[psi.S]*g.P*prob[obj_index][v_true]
                else:
                    factor += (1-accuracy_list[psi.S])*g.P*(1-prob[obj_index][v_true])
        elif g.Oi != obj_index and g.Oj == obj_index and psi.O != obj_index:
            if psi.V == v:
                factor += accuracy_list[psi.S]*g.P*prob[obj_index][v]
            else:
                factor += (1-accuracy_list[psi.S])*g.P*(1-prob[obj_index][v])
    return factor


def get_prob(prob, data, g_data, accuracy_list, obj_index):
    G = g_data[(g_data.Oj == obj_index) | (g_data.Oi == obj_index)]
    Psi = data[data.O.isin(G.Oi.drop_duplicates())]
    possible_values = [0, 1]
    a, b = 1., 1.
    for psi in Psi.iterrows():
        a = get_factor(prob=prob, psi=psi[1], G=G, accuracy_list=accuracy_list, obj_index=obj_index, v=possible_values[0])
        b = get_factor(prob=prob, psi=psi[1], G=G, accuracy_list=accuracy_list, obj_index=obj_index, v=possible_values[1])
    prob = [a/(a+b), b/(a+b)]
    return prob


data = pd.read_csv('../../data/observation_test.csv', names=['S', 'O', 'V'])
accuracy_data = pd.read_csv('../../data/accuracy.csv', names=['S', 'A'])
g_data = pd.read_csv('../../data/g.csv', names=['Oi', 'Oj', 'P'])
truth_obj_list = [0, 1, 1]
observ_val, var_index, accuracy_list, s_number = init_var(data=data, accuracy=accuracy_data)

prob = get_init_prob(data=data)
possible_values = [0, 1]

o_ind = 0
prob[o_ind] = get_prob(prob=prob, data=data, g_data=g_data, accuracy_list=accuracy_list, obj_index=o_ind)

print prob
