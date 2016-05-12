import numpy as np


def get_dist_metric(data, ground_truth, prob):
    truth_obj_val = list(ground_truth.V.values)
    truth_obj_ind = list(ground_truth.O.values)
    prob = [prob[i] for i in truth_obj_ind]
    prob_gt = []
    val = []
    for obj_index in truth_obj_ind:
        possible_values = sorted(data[data.O == obj_index].V.drop_duplicates().values)
        val.append(possible_values)
        l = len(possible_values)
        prob_gt.append([0]*l)
    for obj_ind, v_true in enumerate(truth_obj_val):
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


def get_precision(data, ground_truth, prob):
    truth_obj_val = list(ground_truth.V.values)
    truth_obj_ind = list(ground_truth.O.values)

    obj_result_list = []
    for obj_index in truth_obj_ind:
        possible_values = sorted(data[data.O == obj_index].V.drop_duplicates().values)
        val_ind = prob[obj_index].index(max(prob[obj_index]))
        val = possible_values[val_ind]
        obj_result_list.append(val)
    count = 0
    for gt, res in zip(truth_obj_val, obj_result_list):
        if gt == res:
            count += 1
    precision = float(count)/len(truth_obj_val)

    return precision
