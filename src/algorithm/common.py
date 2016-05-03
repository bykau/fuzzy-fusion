import math
import numpy as np

possible_values = [0, 1]
l = len(possible_values)


def get_log_lik(prob):
    log_l = 0
    for i in prob:
        for j in possible_values:
            if i[j] == 0:
                continue
            log_l += (-1)*math.log(i[j])

    return log_l


def get_dist_metric(data, truth_obj_list, prob):
    prob_gt = []
    val = []
    for obj_index in range(len(truth_obj_list)):
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
