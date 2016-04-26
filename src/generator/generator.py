import pandas as pd
import random
import copy


def generator(cov_list, p_list, ground_truth, claster_size, g_true, values):
    number_of_obj = len(ground_truth)
    source_number = len(p_list)
    while True:
        data = []
        if claster_size-1 != 0:
            g = (1.-g_true)/(claster_size-1)
        for s in range(source_number):
            p_s = p_list[s]
            cov_s = cov_list[s]
            for obj_ind in range(number_of_obj):
                true_val = ground_truth[obj_ind]
                p_val = copy.copy(values)
                if random.uniform(0, 1) <= cov_s:
                    if random.uniform(0, 1) <= g_true:
                        if random.uniform(0, 1) <= p_s:
                            val = true_val
                        else:
                            p_val.remove(true_val)
                            val = random.choice(p_val)
                    else:
                        if obj_ind % 2 == 0:
                            swp_ind = obj_ind + 1
                        else:
                            swp_ind = obj_ind - 1
                        if random.uniform(0, 1) <= p_s:
                            val = ground_truth[swp_ind]
                        else:
                            p_val.remove(ground_truth[swp_ind])
                            val = random.choice(p_val)

                    data.append([s, obj_ind, val])

        g_list = []
        for o_i in range(number_of_obj):
            if o_i % 2 == 0:
                swp_ind = o_i + 1
            else:
                swp_ind = o_i - 1

            for o_j in [o_i, swp_ind]:
                if o_i == o_j:
                    g_list.append([o_i, o_j, g_true])
                else:
                    g_list.append([o_i, o_j, g])

        g_frame = pd.DataFrame(g_list, columns=['Oi', 'Oj', 'P'])
        data_frame = pd.DataFrame(data=data, columns=['S', 'O', 'V'])
        if len(data_frame.O.drop_duplicates()) != number_of_obj:
            print '!'
            number_of_obj
            continue
        break
    return [data_frame, g_frame]
