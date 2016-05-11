from common import get_dist_metric, get_precision


def m_voting(data, truth_obj_list):
    obj_index_list = sorted(data.O.drop_duplicates())
    prob = []
    for obj in obj_index_list:
        obj_values = list(data[data.O == obj].V.values)
        possible_values = sorted(list(data[data.O == obj].V.drop_duplicates().values))
        obj_pr = []
        for v in possible_values:
            v_count = obj_values.count(v)
            obj_pr.append(v_count)
        norm_const = len(obj_values)
        obj_pr = [float(i)/norm_const for i in obj_pr]
        prob.append(obj_pr)
    dist_metric = get_dist_metric(data=data, truth_obj_list=truth_obj_list, prob=prob[0:len(truth_obj_list)])
    precision = get_precision(data=data, truth_obj_list=truth_obj_list, prob=prob[0:len(truth_obj_list)])

    return dist_metric, precision
