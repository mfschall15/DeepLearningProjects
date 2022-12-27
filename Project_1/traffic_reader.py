import pickle
import matplotlib.pyplot as plt
import numpy as np

def load_traffic(path, kind='train', subclass=None):
    import os
    import gzip
    import numpy as np

    t_file = path

    """Load traffic data from `path`"""
    with open(t_file, mode='rb') as f:
        train = pickle.load(f)

    images, labels = train['features'], train['labels']
    images = images.reshape((images.shape[0], -1))

    return images, labels



