import numpy as np

def evaluate_prf1(
    overlap_mat: np.array, 
    err_ang_mat: np.array, 
    err_dist_mat: np.array, 
    num_assoc_mat: np.array, 
    req_overlap: float,
    req_err_ang: float,
    req_err_dist: float,
    req_assoc: int, 
):
    relevant_elements = np.sum((overlap_mat >= req_overlap))
    true_positives = np.sum((overlap_mat >= req_overlap) & (err_ang_mat <= req_err_ang) & (err_dist_mat <= req_err_dist) & (num_assoc_mat >= req_assoc))
    false_positives = np.sum((overlap_mat < req_overlap) & ((num_assoc_mat >= req_assoc)) | # reported loop closure but submaps did not overlap
                             (overlap_mat >= req_overlap) & ((err_ang_mat > req_err_ang) | (err_dist_mat > req_err_dist)) & (num_assoc_mat >= req_assoc)) # reported loop closure but transformation was wrong
    
    # print(f"Relevant elements: {relevant_elements}")
    # print(f"True positives: {true_positives}")
    # print(f"False positives: {false_positives}")
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / relevant_elements if relevant_elements > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1