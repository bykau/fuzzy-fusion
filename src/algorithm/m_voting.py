from common import get_metrics, get_alg_accuracy


def m_voting(data, gt):
    obj_index_list = data.keys()
    prob = {}
    for obj in obj_index_list:
        obj_values = data[obj][1]
        possible_values = sorted(set(obj_values))
        obj_pr = []
        for v in possible_values:
            v_count = obj_values.count(v)
            obj_pr.append(v_count)
        norm_const = len(obj_values)
        obj_pr = [float(i)/norm_const for i in obj_pr]
        prob.update({obj: obj_pr})
    # dist_metric, precision = get_metrics(data=data, gt=gt, prob=prob)
    #
    # return dist_metric, precision

    alg_accuracy = get_alg_accuracy(data=data, gt=gt, belief=prob)
    return alg_accuracy
