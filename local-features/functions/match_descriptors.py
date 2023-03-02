import numpy as np

def ssd(desc1, desc2):
    '''
    Sum of squared differences
    Inputs:
    - desc1:        - (q1, feature_dim) descriptor for the first image
    - desc2:        - (q2, feature_dim) descriptor for the sencond image
    Returns:
    - distances:    - (q1, q2) numpy array storing the squared distance
    '''
    assert desc1.shape[1] == desc2.shape[1]
    # TODO: implement this function please
    pair_wise_dist = np.linalg.norm(desc1[:, None, :] - desc2[None, :, :], axis=-1) # (q1 x q2 x feature_dim --> q1 x q2)
    return pair_wise_dist ** 2

def match_descriptors(desc1, desc2, method = "one_way", ratio_thresh=0.5):
    '''
    Match descriptors
    Inputs:
    - desc1:        - (q1, feature_dim) descriptor for the first image
    - desc2:        - (q2, feature_dim) descriptor for the first image
    Returns:
    - matches:      - (m x 2) numpy array storing the indices of the matches
    '''
    assert desc1.shape[1] == desc2.shape[1]
    distances = ssd(desc1, desc2)
    q1, q2 = desc1.shape[0], desc2.shape[0]
    matches = None
    if method == "one_way": # Query the nearest neighbor for each keypoint in image 1
        # TODO: implement the one-way nearest neighbor matching here
        matches = np.stack((np.arange(q1), np.argmin(distances, axis = 1)), axis = -1)

    elif method == "mutual":
        # TODO: implement the mutual nearest neighbor matching here
        matches = np.stack((np.arange(q1), np.argmin(distances, axis = 1)), axis = -1)
        mutual_mask = np.argmin(distances, axis = 0)[matches[:, 1]] == matches[:, 0]
        matches = matches[mutual_mask]

    elif method == "ratio":
        # TODO: implement the ratio test matching here
        matches = np.stack((np.arange(q1), np.argmin(distances, axis = 1)), axis = -1)
        NN_2 = np.partition(distances, (0, 1), axis = 1)[:, : 2] # get all the first and second neighbors for each kp in img1
        valid_mask = NN_2[:, 0] / NN_2[:, 1] < ratio_thresh 
        matches = matches[valid_mask]
    else:
        raise NotImplementedError
    return matches

