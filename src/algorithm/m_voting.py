from common import get_dist_metric, get_precision


def m_voting(data, ground_truth):
    obj_index_list = sorted(data.O.drop_duplicates())
    prob = {}
    for obj in obj_index_list:
        obj_values = list(data[data.O == obj].V.values)
        possible_values = sorted(list(set(obj_values)))
        obj_pr = []
        for v in possible_values:
            v_count = obj_values.count(v)
            obj_pr.append(v_count)
        norm_const = len(obj_values)
        obj_pr = [float(i)/norm_const for i in obj_pr]
        prob.update({obj: obj_pr})
        print obj

    dist_metric = get_dist_metric(data=data, ground_truth=ground_truth, prob=prob)
    print dist_metric
    precision = get_precision(data=data, ground_truth=ground_truth, prob=prob)
    print precision

    return dist_metric, precision
