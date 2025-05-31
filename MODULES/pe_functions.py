import numpy as np
from sklearn.preprocessing import normalize

def get_1D_encoding(positions, d):
    """
    Create the position encoding for the postions
    :param positions: (m, n) numpy array, 
                m is the number of samples,
                n is the number of features.
    :param d: the desired total dimension of the output for each feature.   
    :return: (m, D//n) numpy array.       
    """
    def get_pos_angle_vec(k):
        return [k / np.power(10000, 2 * (i // 2) / d) for i in range(d)]

    encoded_1D_feature = np.array([get_pos_angle_vec(k) for k in positions])
    encoded_1D_feature[:, 0::2] = np.sin(encoded_1D_feature[:, 0::2]) 
    encoded_1D_feature[:, 1::2] = np.cos(encoded_1D_feature[:, 1::2])
    return encoded_1D_feature

def get_positions(X, D=128):
    """
    Create the positions for the raw data X
    :param X: (m, n) numpy array, 
                m is the number of samples,
                n is the number of features.
    :param D: the desired total dimension of the output.   
    :return: (m, n) numpy array.       
    """
    X_normalized = normalize(X-np.min(X, axis=0), axis=0, norm='max')
    pos = np.zeros(X.shape)
    for i in range(X.shape[1]):
        d = X_normalized[:, i]
        pos[:, i] = d*(D-1)+1
    pos = pos.astype(int)
    return pos

def position_encoding(X, D=128):
    """
    Compute the positional encoding of the input data X
    :param X: (m, n) numpy array, 
                m is the number of samples,
                n is the number of features.
    :param D: the desired total dimension of the output.   
    :return: (m, n, D//n) numpy array.       
    """
    # PE
    if X.ndim != 2:
        X = X.reshape(1, -1)
    pos = get_positions(X, D=D//X.shape[1])
    m, n = X.shape
    d = D//n
    X_encod = np.zeros((m, n, d))
    for i in range(n):
        X_encod[:, i, :] = get_1D_encoding(pos[:, i], d)
    return X_encod