import numpy as np
import os
import matplotlib.pyplot as plt 
from traffic_reader import load_traffic


def traffic_sign(aligned=True):
    if aligned:
        return load_traffic('data', kind='aligned')
    return load_traffic('data', kind='unaligned')

load_data = traffic_sign

def z_score_normalize(X, u = None, std = None):
    """
    Performs z-score normalization on X. 

    f(x) = (x - μ) / σ
        where 
            μ = mean of x
            σ = standard deviation of x

    Parameters
    ----------
    X : np.array
        The data to min-max normalize

    Returns
    -------
        Tuple:
            Transformed dataset with mean 0 and stdev 1
            Computed statistics (mean and stdev) for the dataset to undo z-scoring.
    """
    
    X = (X-u)/std
    norms = np.linalg.norm(X, axis=1)

    X = X/norms[:,None]
    return X

def min_max_normalize(X, _min = None, _max = None):
    """
    Performs min-max normalization on X. 

    f(x) = (x - min(x)) / (max(x) - min(x))

    Parameters
    ----------
    X : np.array
        The data to min-max normalize

    Returns
    -------
        Tuple:
            Transformed dataset with all values in [0,1]
            Computed statistics (min and max) for the dataset to undo min-max normalization.
    """
    
    return (X-np.min(X))/(np.max(X) - np.min(X))

def onehot_encode(y):
    """
    Performs one-hot encoding on y.

    Ideas:
        NumPy's `eye` function

    Parameters
    ----------
    y : np.array
        1d array (length n) of targets (k)

    Returns
    -------
        2d array (shape n*k) with each row corresponding to a one-hot encoded version of the original value.
    """
    # This part takes whole label array, and converts it to one-hot encoded version, where each row is
    # is the one hot encoded version of respective input, 1st row ---> one-hot encoded version of 1st sample
    encoded_labels = np.zeros((y.size, y.max()+1))
    rows = np.arange(y.size)
    encoded_labels[rows, y] = 1
    return encoded_labels


def onehot_decode(y):
    """
    Performs one-hot decoding on y.

    Ideas:
        NumPy's `argmax` function 

    Parameters
    ----------
    y : np.array
        2d array (shape n*k) with each row corresponding to a one-hot encoded version of the original value.

    Returns
    -------
        1d array (length n) of targets (k)
    """
    # this part takes the one hot encoded version, and outputs the argmax in every row, which is the class of each row(sample)
    return np.argmax(y, axis = 0)

def shuffle(dataset):
    """
    Shuffle dataset.

    Make sure that corresponding images and labels are kept together. 
    Ideas: 
        NumPy array indexing 
            https://numpy.org/doc/stable/user/basics.indexing.html#advanced-indexing

    Parameters
    ----------
    dataset
        Tuple containing
            Images (X)
            Labels (y)

    Returns
    -------
        Tuple containing
            Images (X)
            Labels (y)
    """
    X, y = dataset
    shuffler = np.random.permutation(len(X))
    X = X[shuffler]
    y = y[shuffler]
    
    return X, y

def append_bias(X):
    # This function appends the input data with bias, i.e. an Mx1 column of 1s.
    return np.concatenate((np.ones((len(X),1)), X), axis=1)

def generate_minibatches(dataset, batch_size=64):
    X, y = dataset
    l_idx, r_idx = 0, batch_size
    while r_idx < len(X):
        yield X[l_idx:r_idx], y[l_idx:r_idx]
        l_idx, r_idx = r_idx, r_idx + batch_size

    yield X[l_idx:], y[l_idx:]

def generate_k_fold_set(dataset_path, class_a = None, class_b = None, binary = False, k = 5): 
    # Be sure to modify to include train/val/test
    dataset = load_traffic(dataset_path)
    if binary:
        # Extract only images and labeled classified as class-a or class-b and make binary dataset
        index = np.where((dataset[1] == class_a) | (dataset[1] == class_b))
        index = np.array(index)
        binary_dataset = (dataset[0][index].squeeze(), dataset[1][index].squeeze())
        X,y = binary_dataset
        
    else:    
        X, y = dataset
        
    order = np.random.permutation(len(X))

    fold_width = len(X) // k
    # If k = 1, then the first 80% of shuffled data will be train, the next 10% will be validation 
    # and the last 10% will be the test set.
    if k == 1:
        train = X[order[:int(len(X)*0.8)]], y[order[:int(len(X)*0.8)]]
        validation = X[order[int(len(X)*0.8): int(len(X)*0.9)]], y[order[int(len(X)*0.8): int(len(X)*0.9)]]
        test = X[order[int(len(X)*0.9):]], y[order[int(len(X)*0.9):]]
        yield train, validation, test 
        return

    l_idx, v_idx, i_idx = 0, fold_width,  fold_width * 2
    
    # If k > 1 then folds are arranged by using a technique similar to sliding window, each iteration we cuttings from
    # left end and right end are shifted by a fold width for both train, validation and test.
    for i in range(k):
        if i < k-1:
            train = np.concatenate([X[order[:l_idx]], X[order[i_idx:]]]), np.concatenate([y[order[:l_idx]], y[order[i_idx:]]])
            validation = X[order[l_idx:v_idx]], y[order[l_idx:v_idx]]
            test = X[order[v_idx:i_idx]], y[order[v_idx:i_idx]]
        else:
            train =  X[order[fold_width:-fold_width]], y[order[fold_width:-fold_width]]
            validation = X[order[l_idx:v_idx]], y[order[l_idx:v_idx]]
            test = X[order[0:fold_width]], y[order[0:fold_width]]
        yield train, validation, test
        l_idx, v_idx, i_idx = v_idx, v_idx + fold_width, i_idx + fold_width
          
     