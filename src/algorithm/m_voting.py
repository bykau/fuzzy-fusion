from common import get_dist_metric

possible_values = [0, 1]


def m_voting(data, truth_obj_list):
    obj_index_list = sorted(data.O.drop_duplicates())
    prob = []
    for obj in obj_index_list:
        obj_values = data[data.O == obj].V.values
        p_v1 = float(sum(obj_values))/len(obj_values)
        p_v0 = 1-p_v1
        prob.append([p_v0, p_v1])
    dist_metric = get_dist_metric(data=data, truth_obj_list=truth_obj_list, prob=prob[0:len(truth_obj_list)])

    return dist_metric
