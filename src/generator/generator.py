import pandas as pd
import random


def generator(cov_list, p_list, ground_truth, claster_size):
    source_number = 3
    g = 1./(claster_size+1)
    g_true = g*2
    data = []
    for s in range(source_number):
        p_s = p_list[s]
        cov_s = cov_list[s]
        flag = False
        for obj_ind in range(len(ground_truth)):
            true_val = ground_truth[obj_ind]
            if random.uniform(0, 1) <= cov_s:
                if obj_ind < claster_size:
                    if flag:
                        continue
                    if random.uniform(0, 1) <= g_true:
                        if random.uniform(0, 1) <= p_s:
                            val = true_val
                        else:
                            val = 1-true_val
                    else:
                        claster_ind = range(claster_size)
                        claster_ind.remove(obj_ind)
                        swp_index = random.choice(claster_ind)
                        if random.uniform(0, 1) <= p_s:
                            val = ground_truth[swp_index]
                        else:
                            val = 1-ground_truth[swp_index]
                    flag = True
                else:
                    if random.uniform(0, 1) <= p_s:
                        val = true_val
                    else:
                        val = 1-true_val
                data.append([s, obj_ind, val])
    data_frame = pd.DataFrame(data=data, columns=['S', 'O', 'V'])
    return data_frame

