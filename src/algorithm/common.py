import numpy as np


def get_metrics(data, gt, prob):
    dist = 0.
    gt_objects = gt.keys()
    norm_const = len(gt_objects)
    pres_count = 0.
    for obj in gt_objects:
        possible_values = sorted(set(data[obj][1]))
        try:
            gt_val_ind = possible_values.index(gt[obj])
        except ValueError:
            norm_const -= 1
            continue
        obj_prob = prob[obj]
        dist += obj_prob[gt_val_ind]

        obj_ind = obj_prob.index(max(obj_prob))
        if gt_val_ind == obj_ind:
            pres_count += 1
    dist_norm = dist/norm_const
    precision = pres_count/norm_const

    return dist_norm, precision


def get_accuracy_err(acc_truth, acc):
    err = 0.
    for a_t, a in zip(acc_truth, acc):
        err += abs(a_t - a)

    return err


def get_data(data):
    obj_index_list = np.sort(data.O.drop_duplicates())
    data_new = {}
    for obj in obj_index_list:
        data_obj = data[data.O == obj]
        s_obj = list(data_obj.S.values)
        values_obj = list(data_obj.V.values)
        data_new.update({obj: [s_obj, values_obj]})

    return data_new
