import numpy as np

possible_values = [0, 1]


def get_dist_metric(data, truth_obj_list, prob):
    prob_gt = []
    val = []
    l = len(possible_values)
    for obj_index in range(len(data.O.drop_duplicates())):
        val.append(possible_values)
        prob_gt.append([0]*l)
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


def m_voting(data, truth_obj_list):
    obj_index_list = sorted(data.O.drop_duplicates())
    prob = []
    for obj in obj_index_list:
        obj_values = data[data.O == obj].V.values
        p_v1 = float(sum(obj_values))/len(obj_values)
        p_v0 = 1-p_v1
        prob.append([p_v0, p_v1])
    dist_metric = get_dist_metric(data=data, truth_obj_list=truth_obj_list, prob=prob)

    return dist_metric
