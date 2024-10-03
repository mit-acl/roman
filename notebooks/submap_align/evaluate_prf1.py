import numpy as np
import pickle
import hashlib

def evaluate_prf1(
    overlap_mat: np.array, 
    err_ang_mat: np.array, 
    err_dist_mat: np.array, 
    num_assoc_mat: np.array, 
    req_overlap: float,
    req_err_ang: float,
    req_err_dist: float,
    req_assoc: int, 
    oracle_overlap: bool = False,
    ignore_correct_less_overlap: bool = True
):
    relevant_elements = np.sum((overlap_mat >= req_overlap))
    true_positives = np.sum((overlap_mat >= req_overlap) & (err_ang_mat <= req_err_ang) & (err_dist_mat <= req_err_dist) & (num_assoc_mat >= req_assoc))
    if not oracle_overlap:
        if ignore_correct_less_overlap:
            # ignores if overlap is less than required but transformation is correct
            # false_positives = np.sum((overlap_mat == 0.0) & ((num_assoc_mat >= req_assoc)) | # reported loop closure but submaps did not overlap
            #                         (overlap_mat > 0.0) & ((err_ang_mat > req_err_ang) | (err_dist_mat > req_err_dist)) & (num_assoc_mat >= req_assoc)) # reported loop closure but transformation was wrong
            false_positives = np.sum(((overlap_mat < req_overlap) & ((num_assoc_mat >= req_assoc)) &
                                      ((overlap_mat == 0.0) | ((err_ang_mat > req_err_ang) | (err_dist_mat > req_err_dist)))) | # reported loop closure but submaps did not overlap
                                    ((overlap_mat >= req_overlap) & ((err_ang_mat > req_err_ang) | (err_dist_mat > req_err_dist)) & (num_assoc_mat >= req_assoc))) # reported loop closure but transformation was wrong
        else:
            false_positives = np.sum(((overlap_mat < req_overlap) & ((num_assoc_mat >= req_assoc))) | # reported loop closure but submaps did not overlap
                                    ((overlap_mat >= req_overlap) & ((err_ang_mat > req_err_ang) | (err_dist_mat > req_err_dist)) & (num_assoc_mat >= req_assoc))) # reported loop closure but transformation was wrong
    else:
        false_positives = np.sum((overlap_mat >= req_overlap) & ((err_ang_mat > req_err_ang) | (err_dist_mat > req_err_dist)) & (num_assoc_mat >= req_assoc)) # reported loop closure but transformation was wrong
    
    # print(f"Relevant elements: {relevant_elements}")
    # print(f"True positives: {true_positives}")
    # print(f"False positives: {false_positives}")
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / relevant_elements if relevant_elements > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1

def prf1_sweep(pkl_paths, req_overlap=0.5, req_err_ang=3, req_err_dist=1.5):
    loaded = False
    for i, pkl_path in enumerate(pkl_paths):
        
        try:
            pkl_file = open(pkl_path, 'rb')
        except:
            print(f"Error loading {pkl_path}")
            continue
        pickle_data = pickle.load(pkl_file)
        if not loaded:
            loaded = True
            overlap_mat, err_ang_mat, err_dist_mat, num_assoc_mat = pickle_data[:4]
            overlap_mat = np.reshape(overlap_mat, (-1,1))
            err_ang_mat = np.reshape(err_ang_mat, (-1,1))
            err_dist_mat = np.reshape(err_dist_mat, (-1,1))
            num_assoc_mat = np.reshape(num_assoc_mat, (-1,1))
        else:
            om, eam, edm, nam = pickle_data[:4]
            overlap_mat = np.concatenate((overlap_mat, np.reshape(om, (-1,1))), axis=0)
            err_ang_mat = np.concatenate((err_ang_mat, np.reshape(eam, (-1,1))), axis=0)
            err_dist_mat = np.concatenate((err_dist_mat, np.reshape(edm, (-1,1))), axis=0)
            num_assoc_mat = np.concatenate((num_assoc_mat, np.reshape(nam, (-1,1))), axis=0)
            
    total_overlap = np.sum((overlap_mat >= req_overlap))
    total_pairs = len(overlap_mat)
    # print(f"number of overlapping: {total_overlap}")
    # print(f"total number of map pairs: {total_pairs}")
            
    assoc_reqs = np.arange(3, 50)
    precisions = []
    recalls = []
    f1s = []
    for req_assoc in assoc_reqs:
        precision, recall, f1 = evaluate_prf1(overlap_mat, err_ang_mat, err_dist_mat, num_assoc_mat, req_err_ang=req_err_ang, req_err_dist=req_err_dist, req_overlap=req_overlap, req_assoc=req_assoc)
        if precision == 0 and recall == 0 and f1 == 0:
            precision = np.nan
            recall = np.nan
            f1 = np.nan
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    return precisions, recalls, f1s, assoc_reqs

def dir_is_inter_robot_lc(dir_name):
    robots_str = dir_name.split('/')[-1]
    num_underscores = robots_str.count('_')
    if num_underscores != 1 and num_underscores != 3:
        return False
    elif num_underscores == 1:
        return robots_str.split('_')[0] == robots_str.split('_')[1]
    else:
        return robots_str.split('_')[0] == robots_str.split('_')[2] \
            and robots_str.split('_')[1] == robots_str.split('_')[3]
    
def method_to_color(method):
    hash_object = hashlib.sha256()
    hash_object.update(method.encode('utf-8'))
    hex_dig = hash_object.hexdigest()
    seed = (int(hex_dig, 16) + 4) % 2**32
    np.random.seed(seed)
    color = np.random.rand(3)
    return color