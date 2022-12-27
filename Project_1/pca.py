import numpy as np
from traffic_reader import load_traffic


class PCA:
    """
    This class handles all things related to PCA for PA1.

    You can add any new parameters you need to any functions. This is an 
    outline to help you get started.

    You should run PCA on both the training and validation / testing datasets 
    using the same object.

    For the visualization of the principal components, use the internal 
    parameters that are set in `fit`.
    """
    def __init__(self, num_components, normalised_vectors = 0, mean = 0):
        """
        Setup the PCA object. 

        Parameters
        ----------
        num_components : int
            The number of principal components to reduce to.
        """
        self.num_components = num_components
        self.normalised_vectors = normalised_vectors
        self.mean = mean

    def fit(self, X, is_train=False):
        """
        Set the internal parameters of the PCA object to the data.

        Parameters
        ----------
        X : np.array
            Training data to fit internal parameters.
        """
        if is_train==True:
            train_images = X
            k = self.num_components
            mean_train = np.mean(train_images, axis=0)
            # Substracting the mean image from every sample
            mean_reduced = train_images - mean_train
            # Calculating A'A matrix where A is the matrix whose rows are different samples
            A_T_A = np.matmul(np.transpose(mean_reduced), mean_reduced)
            A_T_A = A_T_A / len(train_images)
            eig_values, eig_vectors = np.linalg.eigh(A_T_A) 
            # Sorting the eigenvalues in descending way
            idx = eig_values.argsort()[::-1]   
            eig_values = eig_values[idx]
            eig_vectors = eig_vectors[:,idx]
            # Dividing each eigenvector by their respective lambda^(1/2)
            eig_vectors = eig_vectors / np.sqrt(eig_values)

            k_components = eig_vectors[:,:k]
            # Mean and principal components of the train set will be used in projecting validation and test set.
            self.mean = mean_train
            self.normalised_vectors = k_components
       
        else:
            pass


    def transform(self, X):
        """
        Use the internal parameters set with `fit` to transform data.

        Make sure you are using internal parameters computed during `fit` 
        and not recomputing parameters every time!

        Parameters
        ----------
        X : np.array - size n*k
            Data to perform dimensionality reduction on

        Returns
        -------
            Transformed dataset with lower dimensionality
        """

        # Substracting mean of train set from each sample
        mean_substracted = X - self.mean
        # Projecting them with using the principal components found by using train set
        data_principal_components = np.matmul(mean_substracted, self.normalised_vectors)
        return data_principal_components


    def fit_transform(self, X, is_train=False):
        self.fit(X, is_train)
        return self.transform(X)
