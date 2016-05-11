import numpy as np


def get_dist_metric(data, truth_obj_list, prob):
    prob_gt = []
    val = []
    for obj_index in range(len(truth_obj_list)):
        possible_values = sorted(data[data.O == obj_index].V.drop_duplicates().values)
        val.append(possible_values)
        l = len(possible_values)
        prob_gt.append([0]*l)
    for obj_ind, v_true in enumerate(truth_obj_list):
        for v_ind, v in enumerate(val[obj_ind]):
            if v == v_true:
                prob_gt[obj_ind][v_ind] = 1.
    prob_gt_vector = []
    prob_vector = []
    for i in range(len(prob_gt)):
        prob_gt_vector += prob_gt[i]
        prob_vector += prob[i]
    dist_metric = np.dot(prob_gt_vector, prob_vector)
    dist_metric_norm = dist_metric/len(prob_gt)

    return dist_metric_norm


def get_precision(data, truth_obj_list, prob):
    obj_result_list = []
    for obj_index in range(len(truth_obj_list)):
        possible_values = sorted(data[data.O == obj_index].V.drop_duplicates().values)
        val_ind = prob[obj_index].index(max(prob[obj_index]))
        result_val = possible_values[val_ind]
        obj_result_list.append(result_val)
    count = 0
    for gt, res in zip(truth_obj_list, obj_result_list):
        if gt == res:
            count += 1
    precision = float(count)/len(truth_obj_list)

    return precision