import pandas as pd
import random


def generator(cov_list, p_list, ground_truth, claster_size, g_true):
    number_of_obj = len(ground_truth)
    source_number = len(p_list)
    if claster_size-1 != 0:
        g = (1.-g_true)/(claster_size-1)
    data = []

    while True:
        for s in range(source_number):
            p_s = p_list[s]
            cov_s = cov_list[s]
            for obj_ind in range(number_of_obj):
                true_val = ground_truth[obj_ind]
                if random.uniform(0, 1) <= cov_s:
                    if obj_ind < claster_size:
                        if random.uniform(0, 1) <= g_true:
                            if random.uniform(0, 1) <= p_s:
                                val = true_val
                            else:
                                val = 1-true_val
                        else:
                            claster_ind = range(claster_size)
                            claster_ind.remove(obj_ind)
                            swp_index = claster_ind[random.randint(0, len(claster_ind)-1)]
                            if random.uniform(0, 1) <= p_s:
                                val = ground_truth[swp_index]
                            else:
                                val = 1-ground_truth[swp_index]
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
        if len(data_frame.O.drop_duplicates()) != number_of_obj:
            print '!'
            number_of_obj
            continue
        break
    return [data_frame, g_frame]
