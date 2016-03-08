import pandas as pd
import random


def generator(cov_list, p_list, ground_truth, claster_size):
    number_of_obj = len(ground_truth)
    source_number = 3
    g = 1./(claster_size+1)
    g_true = g*2
    data = []
    for s in range(source_number):
        p_s = p_list[s]
        cov_s = cov_list[s]
        flag = False
        for obj_ind in range(number_of_obj):
            true_val = ground_truth[obj_ind]
            if random.uniform(0, 1) <= cov_s:
                if obj_ind < claster_size:
                    if flag:
                        continue
                    if random.uniform(0, 1) <= g_true:
                        if random.uniform(0, 1) <= p_s:
                            val = ground_truth[s]
                        else:
                            val = 1-ground_truth[s]
                    else:
                        claster_ind = range(claster_size)
                        claster_ind.remove(s)
                        swp_index = random.choice(claster_ind)
                        if random.uniform(0, 1) <= p_s:
                            val = ground_truth[swp_index]
                        else:
                            val = 1-ground_truth[swp_index]
                    flag = True
                    data.append([s, s, val])
                else:
                    if random.uniform(0, 1) <= p_s:
                        val = true_val
                    else:
                        val = 1-true_val
                    data.append([s, obj_ind, val])

    g_list = []
    for o_i in range(number_of_obj):
        if o_i < claster_size:
            for o_j in range(claster_size):
                if o_i == o_j:
                    g_list.append([o_i, o_j, g_true])
                else:
                    g_list.append([o_i, o_j, g])
        else:
            g_list.append([o_i, o_i, 1.])

    g_frame = pd.DataFrame(g_list, columns=['Oi', 'Oj', 'P'])
    data_frame = pd.DataFrame(data=data, columns=['S', 'O', 'V'])
    return [data_frame, g_frame]
