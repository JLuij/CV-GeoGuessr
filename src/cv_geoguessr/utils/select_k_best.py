def select_k_best(data, k):
    return (-data).argsort()[:k]
    