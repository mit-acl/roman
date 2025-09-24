import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import Union

def global_nearest_neighbor(data1: list, data2: list, similarity_fun: callable, similarity_range: np.ndarray = None):
    """
    Associates data1 with data2 using the global nearest neighbor algorithm.

    Args:
        data1 (list): List of first data items
        data2 (list): List of second data items
        similarity_fun (callable(item1, item2)->float|np.ndarray): Evaluates the similarity between two items
        similarity_range (np.ndarray): (2, N) array of threshold and maximum similarity scores

    Returns:
        list of pairs (data1, data2) indicies that should be associated together
    """
    len1 = len(data1)
    len2 = len(data2)
    scores = np.zeros((len1, len2))
    M = 1e9 # just a large number

    for i in range(len1):
        for j in range(len2):
            similarity = similarity_fun(data1[i], data2[j])
            
            # both scores must be above threshold for association
            if np.any(similarity < similarity_range[0, :]):
                score = M
            else:
                if isinstance(similarity, np.ndarray):
                    sim_norm = (similarity - similarity_range[0, :]) / (similarity_range[1, :] - similarity_range[0, :])
                    score = -np.power(np.prod(sim_norm), 1.0/len(similarity)) # geometric mean
                else:
                    score = -similarity # Hungarian is trying to associate low similarity values, so negate
            scores[i,j] = score

    # augment cost to add option for no associations
    hungarian_cost = np.concatenate([
        np.concatenate([scores, np.ones(scores.shape)], axis=1),
        np.ones((scores.shape[0], 2*scores.shape[1]))], axis=0)
    row_ind, col_ind = linear_sum_assignment(hungarian_cost)

    pairs = []
    for idx1, idx2 in zip(row_ind, col_ind):
        # state and measurement associated together
        if idx1 < len1 and idx2 < len2:
            pairs.append((idx1, idx2))

    return pairs