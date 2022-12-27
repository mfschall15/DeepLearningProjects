from cmath import inf
import numpy as np
import data
import time
from numpy.core.umath_tests import inner1d
import time
import pandas as pd
# import tqdm

"""
NOTE
----
Start by implementing your methods in non-vectorized format - use loops and other basic programming constructs.
Once you're sure everything works, use NumPy's vector operations (dot products, etc.) to speed up your network.
"""

def sigmoid(a):
    """
    Compute the sigmoid function.

    f(x) = 1 / (1 + e ^ (-x))

    Parameters
    ----------
    a
        The internal value while a pattern goes through the network
    Returns
    -------
    float
       Value after applying sigmoid (z from the slides).
    """
    b = a[0]
    return 1/(1+np.exp(-b))
    

def softmax(a):
    """
    Compute the softmax function.

    f(x) = (e^x) / Σ (e^x)

    Parameters
    ----------
    a
        The internal value while a pattern goes through the network
    Returns
    -------
    float
       Value after applying softmax (z from the slides).
    """
    y = np.exp(a)
    # a is a matrix whose each column is the vector for respective input
    y = y/y.sum(axis=0)
    # this will produce a y matrix whose each column is the output of respective input, 1st column -> output of 1st pattern etc.
    return y

def binary_cross_entropy(y, t):
    """
    Compute binary cross entropy.

    L(x) = t*ln(y) + (1-t)*ln(1-y)

    Parameters
    ----------
    y
        The network's predictions
    t
        The corresponding targets
    Returns
    -------
    float 
        binary cross entropy loss value according to above definition
    """
    # a value of 0.000001 is added to log() terms in order to prevent log() going to -inf.
    cross_ent = (t * np.log(y + 0.000001) + (1 - t) * np.log(1 - y + 0.000001))
    cross_ent = -np.sum(cross_ent)/len(t)
    
    return cross_ent

def multiclass_cross_entropy(y, t):
    """
    Compute multiclass cross entropy.

    L(x) = - Σ (t*ln(y))

    Parameters
    ----------
    y
        The network's predictions
    t
        The corresponding targets
    Returns
    -------
    float 
        multiclass cross entropy loss value according to above definition
    """
    # Assuming t is matrix of targets where each row is the one-hot encoded target of respective input
    log_labels = np.log(y)
    # y is a matrix whose each column is the output of respective input and t is matrix where each row represents each input.
    # The command below is the most efficient way to calculate the trace of np.matmul(t,log_labels).
    # Note that we only need the components on the diagonal to calculate multiclass cross entropy, no need to store the other elements.
    multiclass_entropy = -np.sum(inner1d(t, log_labels.T))
    return multiclass_entropy
    
class Network:
    def __init__(self, hyperparameters):
        """
        Perform required setup for the network.

        Initialize the weight matrix, set the activation function, save hyperparameters.

        You may want to create arrays to save the loss values during training.

        Parameters
        ----------
        hyperparameters
            A Namespace object from `argparse` containing the hyperparameters
        activation
            The non-linear activation function to use for the network
        loss
            The loss function to use while training and testing
        """
        self.hyperparameters = hyperparameters
        self.logictic = hyperparameters.logistic
        self.batch = hyperparameters.batch
        # Initializing the weights as a vector(or a matrix in softmax regression) of 0.1s.
        self.weights = np.ones((hyperparameters.in_dim + 1, hyperparameters.out_dim)) * .1
        # Optimum weights are created to store the weights at lowest validation loss.
        self.opt_weights = self.weights
        self.min_val_loss = inf

    def forward(self, X, is_logistic):
        """
        Apply the model to the given patterns

        Use `self.weights` and `self.activation` to compute the network's output

        f(x) = σ(w*x)
            where
                σ = non-linear activation function
                w = weight matrix

        Make sure you are using matrix multiplication when you vectorize your code!

        Parameters
        ----------
        X
            Patterns to create outputs for
        """
        f = sigmoid(np.matmul(self.weights.T, X.T))
        if is_logistic == False:
            f = softmax(np.matmul(self.weights.T, X.T))

        return f

    # def __call__(self, X):
    #     return self.forward(X)

    def train(self, train_minibatch, val_minibatch):
        """
        Train the network on the given minibatch

        Use `self.weights` and `self.activation` to compute the network's output
        Use `self.loss` and the gradient defined in the slides to update the network.

        Parameters
        ----------
        minibatch
            The minibatch to iterate over

        Returns
        -------
        Tuple
            average loss over the minibatch
            accuracy over the minibatch
        """
        is_logistic = self.logictic
        batch = self.batch
            
        X, t = train_minibatch
        # If logistic regression is used, train labels are converted into a vector of 0s and 1s.
        if is_logistic:
            t_labels = np.zeros_like(t)
            t_labels[t == self.hyperparameters.first] = 0 
            t_labels[t == self.hyperparameters.second] = 1
           
        val_X, val_t = val_minibatch
        val_labels = np.zeros_like(val_t)
        
        f = self.forward(X, is_logistic)  
        if is_logistic == True:
            train_labels = np.zeros_like(t_labels)
            train_loss = binary_cross_entropy(f, t_labels)
            # Gradient of the loss is calculated and batch gradient descent is applied using the gradient.
            DJ = np.matmul((t_labels - f), X)
            DJ = np.expand_dims(DJ, axis = 1)
            self.weights = self.weights + self.hyperparameters.learning_rate * DJ
            val_predictions = self.forward(val_X, is_logistic)
            # Validation labels are created from validation predictions.
            val_labels[val_predictions > .5] = 1
            train_labels[f > .5] = 1
            # Validation targets are converted into a vector of 0s and 1s to be used in loss calculation.
            val_t[val_t == self.hyperparameters.first] = 0 
            val_t[val_t == self.hyperparameters.second] = 1
            
            train_acc = np.sum(train_labels == t_labels)/len(t_labels)
            val_acc = np.sum(val_labels==val_t)/len(val_t)  
            print('val acc',val_acc)
            print('train acc', train_acc)
            val_loss = binary_cross_entropy(val_predictions, val_t)
        else:
            # If stoichastic gradient descent is used, we shuffle the data and calculate the f again in every epoch.
            if not batch:
                train_shuffler = X, t
                X, t = data.shuffle(train_shuffler)
                f = self.forward(X, is_logistic) 
            t_encoded = data.onehot_encode(t)            
            train_loss = multiclass_cross_entropy(f, t_encoded) 
            # If BGD is used, we use all of the training set at once to calculate gradient and update the weights.             
            diff = t_encoded-np.transpose(f)            
            if batch:               
                DJ = np.matmul(np.transpose(diff),X)                
                DJ = np.transpose(DJ)
                self.weights = self.weights + self.hyperparameters.learning_rate * DJ     
            # If SGD is used, we iterate through a loop of samples and calculate gradient and update the weights at each iteration.          
            else:
                for i in range(len(X)):
                    DJ = np.outer(diff[i],X[i])
                    DJ = np.transpose(DJ)
                    self.weights = self.weights + self.hyperparameters.learning_rate * DJ
            val_predictions = self.forward(val_X, is_logistic)
            val_labels = data.onehot_decode(val_predictions)
            val_acc = np.sum(val_labels==val_t)/len(val_t)
            train_labels = data.onehot_decode(f)
            train_acc = np.sum(train_labels==t)/len(X)
            val_loss = multiclass_cross_entropy(val_predictions, data.onehot_encode(val_t))
        
        # Weights are stored seperetaly when minimum validation loss is achieved.
        if val_loss < self.min_val_loss:
            self.opt_weights = self.weights
            self.min_val_loss = val_loss
        
        return train_loss, val_loss, val_acc, train_acc
        

    def test(self, test_minibatch, best_model=False):
        """
        Test the network on the given minibatch

        Use `self.weights` and `self.activation` to compute the network's output
        Use `self.loss` to compute the loss.
        Do NOT update the weights in this method!

        Parameters
        ----------
        minibatch
            The minibatch to iterate over
        """

        is_logistic = self.logictic
        test_X, test_t = test_minibatch
        test_labels = np.zeros_like(test_t)

        # Use either current weights or the optimum weights
        used_weights = self.weights
        if best_model:
            used_weights = self.opt_weights
            

        if is_logistic:
            test_predictions = sigmoid(np.matmul(used_weights.T,test_X.T))
            test_labels[test_predictions > .5] = 1
            test_t[test_t == self.hyperparameters.first] = 0 
            test_t[test_t == self.hyperparameters.second] = 1
            test_acc = np.sum(test_labels==test_t)/len(test_t)
            test_loss = binary_cross_entropy(test_predictions, test_t)
        else:
            test_predictions = softmax(np.matmul(used_weights.T,test_X.T))
            test_labels = data.onehot_decode(test_predictions)
            test_acc = np.sum(test_labels==test_t)/len(test_t)
            test_loss = multiclass_cross_entropy(test_predictions, data.onehot_encode(test_t))     
            # If best model is used, also return the data to be used in calculation of confusion matrix.    
            if best_model:
                y_actul = pd.Series(test_t, name = 'Actual')
                y_pred = pd.Series(test_labels, name = 'Predicted')
                df_confusion = pd.crosstab(y_actul, y_pred)
                return test_acc, test_loss, df_confusion, self.opt_weights
        return test_acc, test_loss

