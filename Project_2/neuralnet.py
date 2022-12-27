################################################################################
# CSE 251B: Programming Assignment 2
# Winter 2022
################################################################################
# To install PyYaml, refer to the instructions for your system:
# https://pyyaml.org/wiki/PyYAMLDocumentation
################################################################################
# If you don't have NumPy installed, please use the instructions here:
# https://scipy.org/install.html
################################################################################

from cmath import inf
import os, gzip
from matplotlib import image
import yaml
import numpy as np
import pickle
import sys
import matplotlib.pyplot as plt


def load_config(path):
    """
    Load the configuration from config.yaml.
    """
    return yaml.load(open('config.yaml', 'r'), Loader=yaml.SafeLoader)


def normalize_data(inp, mean_rgb=None, std_rgb=None):
    """
    TODO: Normalize your inputs here to have 0 mean and unit variance.
    """
    # Seperates input array by color
    red = inp[:,:1024]
    green = inp[:,1024:2048]
    blue = inp[:,2048:]
    seperate = [red,green,blue]
    out = None

    for ind in range(len(seperate)):
        out_color = (seperate[ind] - mean_rgb[ind])/std_rgb[ind]
        if hasattr(out,'shape'):
            out = np.hstack((out,out_color))
        else:
            out = out_color
    return out


def mean_std_train(x):
    red = x[:,:1024]
    green = x[:,1024:2048]
    blue = x[:,2048:]
    seperate = [red,green,blue]
    means = []
    stds = []
    for color in seperate:
        mean = np.mean(color)
        means.append(mean)
        std = np.std(color)
        stds.append(std)
    return means, stds

def one_hot_encoding(labels, num_classes=10):
    """
    TODO: Encode labels using one hot encoding and return them.
    """
    # This part takes whole label array, and converts it to one-hot encoded version, where each row is
    # is the one hot encoded version of respective input, 1st row ---> one-hot encoded version of 1st sample
    # encoded_labels = np.zeros((y.size, y.max()+1))
    # rows = np.arange(y.size)
    # encoded_labels[rows, y] = 1
    # return encoded_labels

    shape = (labels.size, labels.max() + 1)
    one_hot = np.zeros(shape)
    rows = np.arange(labels.size)
    
    one_hot[rows, labels] = 1
    
    return one_hot


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
    return np.argmax(y, axis = 1)


def generate_minibatches(X, y, batch_size=128):
    l_idx, r_idx = 0, batch_size
    while r_idx < len(X):
        yield X[l_idx:r_idx], y[l_idx:r_idx]
        l_idx, r_idx = r_idx, r_idx + batch_size

    yield X[l_idx:], y[l_idx:]


def shuffle(X,y):
    """
    Shuffle dataset.
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
    
    shuffler = np.random.permutation(len(X))
    X = X[shuffler]
    y = y[shuffler]
    
    return X, y


def load_data(path, mode='train', val_per = .2):
    """
    Load CIFAR-10 data.
    """
    def unpickle(file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    cifar_path = os.path.join(path, "cifar-10-batches-py")

    if mode == "train":
        images = []
        labels = []
        for i in range(1,6):
            images_dict = unpickle(os.path.join(cifar_path, f"data_batch_{i}"))
            data = images_dict[b'data']
            label = images_dict[b'labels']
            labels.extend(label)
            images.extend(data)
        labels = np.array(labels)
        images = np.array(images)
        one_hot_labels    = one_hot_encoding(labels, num_classes=10) #(n,10)     
        images, one_hot_labels = shuffle(images, one_hot_labels)
        
        return images, one_hot_labels
    elif mode == "test":
        test_images_dict = unpickle(os.path.join(cifar_path, f"test_batch"))
        test_data = test_images_dict[b'data']
        test_labels = test_images_dict[b'labels']
        test_data = np.array(test_data)
        test_labels = np.array(test_labels)
        one_hot_labels    = one_hot_encoding(test_labels, num_classes=10) #(n,10)
        
        test_data, one_hot_labels = shuffle(test_data, one_hot_labels)
        return test_data, one_hot_labels
    else:
        raise NotImplementedError(f"Provide a valid mode for load data (train/test)")    


def substract_val(X, y, val_per = 0.2):
    n = len(X)
    train_images = X[:int((1 - val_per) * n)]
    val_images = X[int((1 - val_per) * n):]
    train_labels = y[:int((1 - val_per) * n),:]
    val_labels = y[int((1 - val_per) * n):,:]    
    return train_images, train_labels, val_images, val_labels


def softmax(a):
    """
    TODO: Implement the softmax function here.
    Remember to take care of the overflow condition.
    """
    y = np.exp(np.transpose(a))
    # a is a matrix whose each row is the vector for respective input
    y = y/y.sum(axis=0)
    # this will produce a y matrix whose each column is the output of respective input, 1st column -> output of 1st pattern etc.
    # so we return the transpose of it where each row is the output of respective input.
    return np.transpose(y)

    # raise NotImplementedError("Softmax not implemented")


class Activation():
    """
    The class implements different types of activation functions for
    your neural network layers.

    Example (for sigmoid):
        >>> sigmoid_layer = Activation("sigmoid")
        >>> z = sigmoid_layer(a)
        >>> gradient = sigmoid_layer.backward(delta=1.0)
    """

    def __init__(self, activation_type = "sigmoid"):
        """
        TODO: Initialize activation type and placeholders here.
        """
        if activation_type not in ["sigmoid", "tanh", "ReLU", "leakyReLU"]:
            raise NotImplementedError(f"{activation_type} is not implemented.")

        # Type of non-linear activation.
        self.activation_type = activation_type

        # Placeholder for input. This will be used for computing gradients.
        self.x = None

    def __call__(self, a):
        """
        This method allows your instances to be callable.
        """
        return self.forward(a)

    def forward(self, a):
        """
        Compute the forward pass.
        """
        self.x = a
        if self.activation_type == "sigmoid":
            return self.sigmoid(a)

        elif self.activation_type == "tanh":
            return self.tanh(a)

        elif self.activation_type == "ReLU":
            return self.ReLU(a)

        elif self.activation_type == "leakyReLU":
            return self.leakyReLU(a)

    def backward(self, delta):
        """
        Compute the backward pass.
        """
        if self.activation_type == "sigmoid":
            grad = self.grad_sigmoid()

        elif self.activation_type == "tanh":
            grad = self.grad_tanh()

        elif self.activation_type == "ReLU":
            grad = self.grad_ReLU()
            
        elif self.activation_type == "leakyReLU":
            grad = self.grad_leakyReLU()

        return grad * delta

    def sigmoid(self, x):
        """
        TODO: Implement the sigmoid activation here.
        """
        return 1/(1+np.exp(-x))

    def tanh(self, x):
        """
        TODO: Implement tanh here.
        """
        return np.tanh(x)
        # raise NotImplementedError("Tanh not implemented")

    def ReLU(self, x):
        """
        TODO: Implement ReLU here.
        """
        return np.maximum(0,x)

    def leakyReLU(self, x):
        """
        TODO: Implement leaky ReLU here.
        """
        return np.where(x > 0, x, x * 0.1)

    def grad_sigmoid(self):
        """
        TODO: Compute the gradient for sigmoid here.
        """
       
        return self.sigmoid(self.x) * (1 - self.sigmoid(self.x))

    def grad_tanh(self):
        """
        TODO: Compute the gradient for tanh here.
        """
        return 1 - np.tanh(self.x) * np.tanh(self.x)
        # raise NotImplementedError("tanh gradient not implemented")

    def grad_ReLU(self):
        """
        TODO: Compute the gradient for ReLU here.
        """
        
        gradient = self.x.copy()
        gradient[gradient < 0] = 0
        gradient[gradient > 0] = 1
        
        return gradient

    def grad_leakyReLU(self):
        """
        TODO: Compute the gradient for leaky ReLU here.
        """
        
        gradient = self.x.copy()
      
        gradient[gradient > 0] = 1
        gradient[gradient < 0] = .1
        
        return gradient


class Layer():
    """
    This class implements Fully Connected layers for your neural network.

    Example:
        >>> fully_connected_layer = Layer(784, 100)
        >>> output = fully_connected_layer(input)
        >>> gradient = fully_connected_layer.backward(delta=1.0)
    """

    def __init__(self, in_units, out_units):
        """
        Define the architecture and create placeholder.
        """
        np.random.seed(42)
        self.w = np.random.randn(in_units, out_units) * .01    # Declare the Weight matrix
        self.b = np.random.randn(1,out_units) * .01 # Create a placeholder for Bias
        self.x = None    # Save the input to forward in this
        self.a = None    # Save the output of forward pass in this (without activation)

        self.d_x = None  # Save the gradient w.r.t x in this
        self.d_w = None  # Save the gradient w.r.t w in this
        self.d_b = None  # Save the gradient w.r.t b in this

    def __call__(self, x):
        """
        Make layer callable.
        """
        return self.forward(x)

    def forward(self, x):
        """
        TODO: Compute the forward pass through the layer here.
        DO NOT apply activation here.
        Return self.a
        """
        self.x = x
        # from d-dim input x to h-dim 'a' vector
        # assuming self.x is a nxd matrix, the following line should output nxh 'a' matrix
        self.a = np.matmul(self.x, self.w) + self.b
        return self.a

        # raise NotImplementedError("Layer forward pass not implemented.")

    def backward(self, delta):
        """
        TODO: Write the code for backward pass. This takes in gradient from its next layer as input,
        computes gradient for its weights and the delta to pass to its previous layers.
        Return self.dx
        """
        # assumin delta is 1xc row vector, the following command will produce the delta of this layer, 1xh vector.
        # note that this is not the actual delta, we are missing derivative of activation func.
        self.d_x = np.matmul(delta, np.transpose(self.w))
        # x is nxd, delta is nxc then the following line will produce dxc matrix
        self.d_w = -np.matmul(np.transpose(self.x), delta)/(10)
        self.d_b = -np.sum(delta, axis=0)/(10)
        return self.d_x
        # raise NotImplementedError("Backprop for Layer not implemented.")



class Neuralnetwork():
    """
    Create a Neural Network specified by the input configuration.

    Example:
        >>> net = NeuralNetwork(config)
        >>> output = net(input)
        >>> net.backward()
    """

    def __init__(self, config):
        """
        Create the Neural Network using config.
        """
        self.layers = []     # Store all layers in this list.
        self.x = None        # Save the input to forward in this
        self.y = None        # Save the output vector of model in this
        self.targets = None  # Save the targets in forward in this variable
        self.lam = float(config['L2_penalty'])

        # Add layers specified by layer_specs.
        for i in range(len(config['layer_specs']) - 1):
            self.layers.append(Layer(config['layer_specs'][i], config['layer_specs'][i+1]))
            if i < len(config['layer_specs']) - 2:
                self.layers.append(Activation(config['activation']))

    def __call__(self, x, targets=None):
        """
        Make NeuralNetwork callable.
        """
        return self.forward(x, targets)

    def forward(self, x, targets=None):
        """
        TODO: Compute forward pass through all the layers in the network and return it.
        If targets are provided, return loss as well.
        """
        self.x = x
        layer_output = self.layers[0](self.x)
        for layer in self.layers[1:]:
            layer_output = layer(layer_output)
        self.y = softmax(layer_output)

        if hasattr(targets,'shape'):
            self.targets = targets
            return self.loss(self.y, targets), self.y
        else:
            return self.y 
        # raise NotImplementedError("Forward not implemented for NeuralNetwork")

    def loss(self, logits, targets):
        '''
        TODO: compute the categorical cross-entropy loss and return it.
        '''
        # Assuming t is matrix of targets where each row is the one-hot encoded target of respective input, and y is the output
        # of the network where each row represents 1 pattern.
        log_labels = np.log(logits)
        n = len(targets)
        multiclass_entropy = -np.sum(np.multiply(log_labels, targets)) / (n)
        
        return multiclass_entropy
        # raise NotImplementedError("Loss not implemented for NeuralNetwork")


    def backward(self):
        '''
        TODO: Implement backpropagation here.
        Call backward methods of individual layers.
        '''
        delta_values = []
        # For output layer delta = target - prediction
        delta_layer = self.targets - self.y
        delta_values.append(delta_layer)
        for ind in reversed(range(len(self.layers))):
            delta_layer = self.layers[ind].backward(delta_layer)
            delta_values.append(delta_layer)

        # note that this deltas are reversed, 1st entry for last layer, last entry for the layer above input layer.
        # so we need to reverse them.
        return reversed(delta_values)


        # raise NotImplementedError("Backprop not implemented for NeuralNetwork")


def calc_accuracy(predictions, targets):
    predicted_labels = onehot_decode(predictions)
    target_labels = onehot_decode(targets)
    acc = np.sum(predicted_labels==target_labels)/len(target_labels)
    return acc


def train(model, x_train, y_train, x_valid, y_valid, config):
    """
    TODO: Train your model here.
    Implement batch SGD to train the model.
    Implement Early Stopping.
    Use config to set parameters for training like learning rate, momentum, etc.
    """
    epochs = config['epochs']
    learning_rate = config['learning_rate']
    batch_size = config['batch_size']
    early_stop = config['early_stop']
    early_stop_epoch = config['early_stop_epoch']
    momentum = config['momentum']
    momentum_gamma = config['momentum_gamma']
    L2_pen = float(config['L2_penalty'])
    min_val_loss = inf
    performance_data = {'Train_Losses' : [],
                        'Train_Accs' : [],
                        'Validation_Losses' : [], 
                        'Validation_Accs' : []}
    counter = 0
    first_time = True
    old_weight_change = []
    old_bias_change = []
    for ind in range(0,len(model.layers)):
        # For momentum, we need to keep track of previous weight changes, the arrays for that is created in here.
        if isinstance(model.layers[ind],Layer):
            old_weight_change.append(np.zeros_like(model.layers[ind].w))
            old_bias_change.append(np.zeros_like(model.layers[ind].b))
        else:
            old_weight_change.append(0)
            old_bias_change.append(0)   
    for e in range(epochs):
        x_train, y_train = shuffle(x_train, y_train)
        batches = generate_minibatches(x_train, y_train, batch_size)
        train_loss_list = []
        train_acc_list = []
        for b in batches:
            minibatch_x, minibatch_y = b
            train_loss, train_predictions = model(minibatch_x, minibatch_y)
            train_acc = calc_accuracy(train_predictions, minibatch_y)
            train_loss_list.append(train_loss)
            train_acc_list.append(train_acc)
            train_delta_values = model.backward()

            for ind in reversed(range(0,len(model.layers))):
                # Here we check if the class is Layer or not, and if layer, we apply the momentum gradient descent
                # to the weights.
                if isinstance(model.layers[ind],Layer):
                    weight_change = momentum_gamma*old_weight_change[ind] + (1- momentum_gamma)*model.layers[ind].d_w
                    bias_change = momentum_gamma*old_bias_change[ind] + (1- momentum_gamma)*model.layers[ind].d_b
                    model.layers[ind].w -=  learning_rate * weight_change + (2 * L2_pen * model.layers[ind].w)
                    model.layers[ind].b -=  learning_rate * bias_change + (2 * L2_pen * model.layers[ind].b)
                    old_weight_change[ind] = weight_change
                    old_bias_change[ind] = bias_change
        first_time = True
        # we take the average of the losses of different minibatches.
        train_total_loss = np.mean(np.array(train_loss_list))
        train_total_acc = np.mean(np.array(train_acc_list))
        val_loss, val_predictions = model(x_valid, y_valid)
        val_acc = calc_accuracy(val_predictions, y_valid)
        print("Epoch: {}".format(e))
        print("Validation Acc: {}      Validation Loss: {}".format(val_acc, val_loss))
        print("Train Acc: {}      Train Loss: {}".format(train_total_acc, train_total_loss))
        performance_data['Train_Accs'].append(train_total_acc)
        performance_data['Train_Losses'].append(train_total_loss)
        performance_data['Validation_Losses'].append(val_loss)
        performance_data['Validation_Accs'].append(val_acc)

        # Early stopping is implemented below.
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            counter = 0
        else:
            counter += 1

        if counter == early_stop_epoch and early_stop == True:
            break

    return model, performance_data 

    # raise NotImplementedError("Train method not implemented")


def test(model, X_test, y_test):
    """
    TODO: Calculate and return the accuracy on the test set.
    """
    test_loss, test_predictions = model(X_test, y_test)
    test_acc = calc_accuracy(test_predictions, y_test)
    return test_loss, test_acc
    
    # raise NotImplementedError("Test method not implemented")

def plotter(performance_data, test_loss, test_acc):
    epochs = np.arange(len(performance_data["Train_Losses"])) + 1
    plt.figure()
    plt.plot(epochs, performance_data["Train_Losses"], label = 'Train Loss')
    plt.plot(epochs, performance_data["Validation_Losses"], label = 'Validation Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(loc = 'upper right')
    plt.show()
    
    plt.figure()
    plt.plot(epochs, np.array(performance_data["Train_Accs"])*100, label = 'Train Accuracy')
    plt.plot(epochs, np.array(performance_data["Validation_Accs"])*100, label = 'Validation Accuracy')
    plt.ylabel('Accuracy Percentage')
    plt.xlabel('Epochs')
    plt.legend(loc = 'upper left')
    plt.show()
    
    return

def partb(model, x, y):
    eps = 10e-2
    # do 1 forward pass and 1 backward pass.
    temp1, temp2 = model(x,y)
    delta_temps = model.backward()
    layer_num = 1 
    for layer in model.layers:
        if isinstance(layer, Layer):
            print(layer.w.shape)
            # check for E(w+e)
            layer.w[0][0] += eps
            w1_loss_up, temp = model(x,y)
            # check for E(w-e)
            layer.w[0][0] -= 2*eps
            w1_loss_down, temp = model(x,y)
            # since we only check 1 weight at a time, we need to convert it back to its original value
            layer.w[0][0] += eps
            # check the difference between numerical approx. and actual gradient.
            numerical_w1 = (w1_loss_up - w1_loss_down)/(2*eps)
            layer.w[0][1] += eps
            w2_loss_up, temp = model(x,y)
            layer.w[0][1] -= 2*eps
            w2_loss_down, temp = model(x,y)
            layer.w[0][1] += eps
            numerical_w2 = (w2_loss_up - w2_loss_down)/(2*eps)
            layer.b[0][0] += eps
            bias_loss_up, temp = model(x, y)
            layer.b[0][0] -= 2*eps
            bias_loss_down, temp = model(x,y)
            layer.b[0][0] += eps
            numerical_bias = (bias_loss_up - bias_loss_down)/(2*eps)
            print("Layer - {}".format(layer_num))
            print("Weight 1 actual gradient: {}     numerical approximation:{}".format(layer.d_w[0][0], numerical_w1))
            print("Weight 2 actual gradient: {}     numerical approximation:{}".format(layer.d_w[0][1], numerical_w2))
            print("Bias actual gradient: {}     numerical approximation:{}".format(layer.d_b[0], numerical_bias))
            layer_num += 1
            
              
if __name__ == "__main__":
    # Load the configuration.
    config = load_config("./")

    # Create the model
    model  = Neuralnetwork(config)

    # Load the data
    x, y = load_data(path="./data", mode="train")
    x_train, y_train, x_valid, y_valid = substract_val(x,y)
    x_test,  y_test  = load_data(path="./data", mode="test")
    mean_train, std_train = mean_std_train(x_train)
    x_train = normalize_data(x_train, mean_train, std_train)
    x_valid = normalize_data(x_valid, mean_train, std_train)
    x_test = normalize_data(x_test, mean_train, std_train)

    # If you want to check part-b uncomment the following lines:
    ##
    #gets data from 1 of each different category
    # part_b_idx = []
    # for i in range(10):
    #     part_b_idx.append(np.where(y[:,i] == 1)[0][0])  
    # part_b_x = x[part_b_idx]
    # part_b_y = y[part_b_idx]
    
    # #Runs Part b
    # partb(model, part_b_x, part_b_y)
    # sys.exit()
    ##

    # Train the model
    model, performance_data = train(model, x_train, y_train, x_valid, y_valid, config)
    # Test the model
    test_loss, test_acc = test(model, x_test, y_test)
    print("Test Acc: {}      Test Loss: {}".format(test_acc, test_loss))

    plotter(performance_data, test_loss, test_acc)

