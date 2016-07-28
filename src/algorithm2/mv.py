from collections import defaultdict


def majority_voting(Psi):
    """
    Majority voting fusing: a true value is the one which got most votes.

    :param Psi: observations (a list of object votes)
    :return: for each object we return a dict in a list. Each key in the dict is an object value, the dict values are
    probs.
    """
    res = []
    for x in Psi:
        counts = defaultdict(int)
        total = 0.0
        for s, val in x:
            counts[val] += 1
            total += 1

        # normalize
        for val in counts.keys():
            counts[val] /= total

        res.append(counts)

    return res
