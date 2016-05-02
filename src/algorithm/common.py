import math
possible_values = [0, 1]


def get_log_lik(prob):
    log_l = 0
    for i in prob:
        for j in possible_values:
            if i[j] == 0:
                continue
            log_l += (-1)*math.log(i[j])

    return log_l
