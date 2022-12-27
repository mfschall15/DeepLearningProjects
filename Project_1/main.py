import argparse
from xmlrpc.client import boolean
import network
import data
from pca import PCA
import numpy as np
import traffic_reader
import time
from tqdm import tqdm
import seaborn as sn
import matplotlib.pyplot as plt


def main(hyperparameters):
        data_path = 'data/train_wb_' + hyperparameters.data + '.p'
        folded = data.generate_k_fold_set(data_path, class_a = hyperparameters.first, class_b = hyperparameters.second, binary = hyperparameters.logistic, k = hyperparameters.k_folds)
        performance_data = []
        # A list of dictionaries is created to store relevant performance data in each fold.
        for i in range(hyperparameters.k_folds):
                performance_data.append({'Train_Losses' : [],
                                         'Train_Accs' : [],
                                        'Validation_Losses' : [], 
                                        'Validation_Accs' : [], 
                                        'Test_Losses' : [], 
                                        'Test_Accs': []})
        fold_num = 0
        for fold in tqdm(folded):
                train_folded, val_folded, test_folded = fold
                train_samples, train_targets = train_folded
                
                val_samples, val_targets = val_folded
                test_samples, test_targets = test_folded

                if train_samples.shape[0] < hyperparameters.in_dim:
                    hyperparameters.in_dim = train_samples.shape[0] -1

                # PCA is applied if pca parameter is True
                if hyperparameters.pca:
                        pca_calculator = PCA(hyperparameters.in_dim)
                        train_samples = pca_calculator.fit_transform(train_samples, is_train=True)
                        val_samples = pca_calculator.fit_transform(val_samples, is_train=False)
                        test_samples = pca_calculator.fit_transform(test_samples, is_train=False)
                # Othervise z-score normalization is applied
                else:
                        test_samples = data.z_score_normalize(test_samples, np.mean(train_samples), np.std(train_samples))
                        val_samples = data.z_score_normalize(val_samples, np.mean(train_samples), np.std(train_samples))
                        train_samples = data.z_score_normalize(train_samples, np.mean(train_samples), np.std(train_samples))

                # After normalization, bias is added to dataset.
                train_samples = data.append_bias(train_samples)
                val_samples = data.append_bias(val_samples)
                test_samples = data.append_bias(test_samples)
                train_minibatch = train_samples, train_targets
                val_minibatch = val_samples, val_targets
                test_minibatch = test_samples, test_targets

                neural_network = network.Network(hyperparameters)
                
                for epoch in tqdm(range(hyperparameters.epochs)):
                        train_loss, val_loss, val_acc, train_acc = neural_network.train(train_minibatch, val_minibatch)
                        performance_data[fold_num]['Train_Accs'].append(train_acc)
                        performance_data[fold_num]['Train_Losses'].append(train_loss)
                        performance_data[fold_num]['Validation_Losses'].append(val_loss)
                        performance_data[fold_num]['Validation_Accs'].append(val_acc)
                        test_acc, test_loss = neural_network.test(test_minibatch)
                        performance_data[fold_num]['Test_Losses'].append(test_loss)
                        performance_data[fold_num]['Test_Accs'].append(test_acc)
                if hyperparameters.logistic == False:
                        best_test_acc, best_test_loss, confusion, weights = neural_network.test(test_minibatch, best_model=True)
                else:
                        best_test_acc, best_test_loss = neural_network.test(test_minibatch, best_model=True)
                fold_num += 1

        return performance_data, confusion, weights

# This funtion does the plotting by using the performance stats from the main function.
def plotter(hyperparameters, performance_stats):
        best_test_acc = 0
        best_test_loss = 0
        average_train_accs = np.zeros(hyperparameters.epochs)
        average_val_accs = np.zeros(hyperparameters.epochs)
        average_test_accs = np.zeros(hyperparameters.epochs)
        average_val_loss = np.zeros(hyperparameters.epochs)
        average_train_loss = np.zeros(hyperparameters.epochs)
        average_test_loss = np.zeros(hyperparameters.epochs)

        for i in range(hyperparameters.k_folds):
                valid_min = np.argmin(performance_stats[i]['Validation_Losses'])
                best_test_acc += performance_stats[i]['Test_Accs'][valid_min]  
                best_test_loss += performance_stats[i]['Test_Losses'][valid_min]
                average_train_accs += performance_stats[i]['Train_Accs']
                average_val_accs += performance_stats[i]['Validation_Accs']
                average_test_accs += performance_stats[i]['Test_Accs']
                average_val_loss += performance_stats[i]['Validation_Losses']
                average_train_loss += performance_stats[i]['Train_Losses']
                average_test_loss += performance_stats[i]['Test_Losses']
    
        best_test_acc = best_test_acc/hyperparameters.k_folds
        best_test_loss = best_test_loss/hyperparameters.k_folds
        average_train_accs = average_train_accs/hyperparameters.k_folds
        average_val_accs = average_val_accs/hyperparameters.k_folds
        average_test_accs = average_test_accs/hyperparameters.k_folds
        average_val_loss = average_val_loss/hyperparameters.k_folds
        average_train_loss = average_train_loss/hyperparameters.k_folds
        average_test_loss = average_test_loss/hyperparameters.k_folds


        val_std = np.zeros(hyperparameters.epochs)
        train_std = np.zeros(hyperparameters.epochs)
        for i in range(50, hyperparameters.epochs+1, 50):
                val_std[i-1] = np.std(average_val_loss[i-50:i])
                train_std[i-1] = np.std(average_train_loss[i-50:i])
    
        print('min test error according to holdout', best_test_loss)
        print('best test acc according to holdout', best_test_acc)
        epochs = np.arange(hyperparameters.epochs)
        plt.figure()
        plt.errorbar(epochs, average_val_loss, label = 'Validation Loss', yerr = val_std, elinewidth = 5)
        plt.errorbar(epochs, average_train_loss, label = 'Train Loss', yerr = train_std)
        plt.ylabel('Loss')
        plt.xlabel('Epochs')
        plt.legend(loc = 'upper right')
        plt.show()
        plt.figure()
        plt.plot(epochs, average_val_accs, label = 'Validation Accuracy')
        plt.plot(epochs, average_train_accs, label = 'Train Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epochs')
        plt.legend(loc = 'upper right')
        plt.show()


parser = argparse.ArgumentParser(description = 'CSE251B PA1')
parser.add_argument('--batch-size', type = int, default = 128,
        help = 'input batch size for training (default: 128)')
parser.add_argument('--epochs', type = int, default = 300,
        help = 'number of epochs to train (default: 300)')
parser.add_argument('--learning-rate', type = float, default = 0.002,
        help = 'learning rate (default: 0.002)')
# parser.add_argument('--z-score', dest = 'normalization', action='store_const',
#         default = data.min_max_normalize, const = data.z_score_normalize,
#         help = 'use z-score normalization on the dataset, default is min-max normalization')
parser.add_argument('--in-dim', type = int, default = 32*32, 
        help = 'number of principal components to use')
parser.add_argument('--out-dim', type = int, default = 43,
        help = 'number of outputs')
parser.add_argument('--k-folds', type = int, default = 1,
        help = 'number of folds for cross-validation default 10')
parser.add_argument('--pca', type = bool, default = False,
        help = 'apply PCA or not (default: True), if True will use # of dims specified in --in-dim')
parser.add_argument('--logistic', type = bool, default = False,
        help = 'True if logistic regression, false if softmax regression')
parser.add_argument('--batch', type = bool, default = True,
        help = 'True if batch gradient descent, false if stoichastic')
parser.add_argument('--data', type = str, default = "aligned",
        help = 'choose aligned or unalighned data, default is aligned')
parser.add_argument('--first', type = int, default = 19,
        help = 'if binary choose first class')
parser.add_argument('--second', type = int, default = 20,
        help = 'if binary choose second class')

hyperparameters = parser.parse_args()
print(hyperparameters)
performance_stats, cm, weights = main(hyperparameters)
plotter(hyperparameters, performance_stats)

if hyperparameters.logistic == False and hyperparameters.batch == True:
        #Calculates confusion matrix percentages and plots it
        cm_per = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        cm_per = np.round(cm_per, decimals = 1)

        fig, ax = plt.subplots(figsize = (20,14))
        sn.heatmap(cm_per, annot = True)
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top') 
        plt.show()

if hyperparameters.logistic == False and hyperparameters.batch == True:
        #Visualizes the weights
        yeet = [3,23,31,37]
        vis = np.ones((1024,4))
        for i, y in enumerate(yeet):
                vis[:,i] = np.interp(weights[1:,y], (weights[1:,y].min(), weights[1:,y].max()), (0, 255))
                a = vis[:,i].reshape((-1,1))
                a = np.reshape(a,(32,32))
                plt.figure()
                plt.title(str(y))
                plt.imshow(a)

