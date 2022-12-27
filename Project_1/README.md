# Performance of Logistic Regression and Softmax Regression with Different Settings on Classifying Traffic Signs
This projet demonstrates the use of logistic and softmax regression on classifying different traffic within the german traffic sign recognition benchmark. I learned about techniques such as PCA, k-folds, Stochastic gradient decent, and other hyperparameter tuning techniques

# How to run the code for Programming Assignment 1:
In order to run the whole code for programming assignment 1, user should run main.py with various arguments. All of those arguments have already default values but to run specific parts, user may need to give different values than the defaults. After running main.py with the desired arguments, the code will automatically produce the required graphs for train and validation accuracies, losses, and also will print test accuracy and loss. 

NOTE: In order for the code to work, there should be a data folder in the same folder with the code and data files should be in it.

# List of arguments:
—epochs : number of epochs to train (default: 300)
—learning-rate: learning rate (default: 0.002)
—in-dim: number of principal components to use (default: 1024)
—out-dim: number of outputs (default: 43)
—k-folds: number of folds for cross-validation (default: 1)
—pca: apply PCA or not (default: True)
—logistic: apply logistic regression if true, softmax regression otherwise (default: True)
—batch: apply BGD if true, SGD if false (default: True)
—data: dataset type (default: aligned)
—first: first class if logistic regression (default: 7)
—second: second class if logistic regression (default: 8)

# Example command:

python main.py —epochs: 100 —out-dim: 1 —pca: True —logistic: True —data: aligned —first: 7 —second: 8

This command will run the code for 100 epochs and will use logistic regression on aligned data with PCA to classify class 7 and 8