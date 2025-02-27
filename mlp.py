import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple


def batch_generator(train_x, train_y, batch_size):
    """
    Generator that yields batches of train_x and train_y.

    :param train_x (np.ndarray): Input features of shape (n, f).
    :param train_y (np.ndarray): Target values of shape (n, q).
    :param batch_size (int): The size of each batch.

    :return tuple: (batch_x, batch_y) where batch_x has shape (B, f) and batch_y has shape (B, q). The last batch may be smaller.
    """
    n = train_x.shape[0]
    for i in range(0, n, batch_size):
        batch_x = train_x[i:i + batch_size]
        batch_y = train_y[i:i + batch_size]
        yield batch_x, batch_y

class ActivationFunction(ABC):
    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the output of the activation function, evaluated on x

        Input args may differ in the case of softmax

        :param x (np.ndarray): input
        :return: output of the activation function
        """
        pass

    @abstractmethod
    def derivative(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the derivative of the activation function, evaluated on x
        :param x (np.ndarray): input
        :return: activation function's derivative at x
        """
        pass

class Sigmoid(ActivationFunction):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return np.exp(-x) / (1+np.exp(-x))**2

class Tanh(ActivationFunction):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return 1 - np.tanh(x) ** 2

class Relu(ActivationFunction):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)
    
    def derivative(self, x: np.ndarray) -> np.ndarray:
        return np.where(x >= 0, 1, 0)

class Softmax(ActivationFunction):
    def forward(self, x: np.ndarray) -> np.ndarray:
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)
    
    def derivative(self, x: np.ndarray) -> np.ndarray:
        s = self.forward(x)
        ssn, fsn = s.shape
        jacobian =  np.zeros((ssn, fsn, fsn))
        for i in range(ssn):
            s_i = s[i].reshape(-1,1)
            jacobian[i] = np.diagflat(s[i]) - np.dot(s_i, s_i.T)
        return jacobian

class Linear(ActivationFunction):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return x
    
    def derivative(self, x: np.ndarray) -> np.ndarray:
        return np.ones_like(x)
        
class Softplus(ActivationFunction):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.log(1 + np.exp(x))
    
    def derivative(self, x: np.ndarray) -> np.ndarray:
        return np.exp(x)/(1 + np.exp(x))

class Mish(ActivationFunction):
    def forward(self, x: np.ndarray) -> np.ndarray:
        softplus = np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)
        return x * softplus
    
    def derivative(self, x: np.ndarray) -> np.ndarray:
        softplus = np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)
        tanh_sp = np.tanh(softplus)
        sigmoid_x = 1 / (1 + np.exp(-x))
        return tanh_sp + x * (1 - tanh_sp**2) * sigmoid_x

class LossFunction(ABC):
    @abstractmethod
    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        pass

class SquaredError(LossFunction):
    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return 0.5 * np.sum((y_true - y_pred)**2, axis=1)
    
    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return (y_pred - y_true) / y_true.shape[0]

class CrossEntropy(LossFunction):
    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return -np.sum(y_true * np.log(y_pred), axis=1)

    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return y_pred - y_true

class Layer:
    def __init__(self, fan_in: int, fan_out: int, activation_function: ActivationFunction, dropout_rate: float = 0.0):
        """
        Initializes a layer of neurons

        :param fan_in: number of neurons in previous (presynpatic) layer
        :param fan_out: number of neurons in this layer
        :param activation_function: instance of an ActivationFunction
        """
        self.fan_in = fan_in
        self.fan_out = fan_out
        self.activation_function = activation_function
        self.dropout_rate = dropout_rate
        self.dropout_mask = None
        self.is_training = True
        
        # He initialization for ReLU-like activations
        if isinstance(activation_function, (Relu, Softplus, Mish)):
            self.W = np.random.randn(fan_in, fan_out) * np.sqrt(2.0 / fan_in)
        else:
            limit = np.sqrt(6 / (fan_in + fan_out))
            self.W = np.random.uniform(-limit, limit, (fan_in, fan_out))
        self.b = np.zeros((1, fan_out))
        self.z = None
        
    def forward(self, h: np.ndarray):
        """
        Computes the activations for this layers

        :param h: input to layer
        :return: layer activations
        """
        weighted_sum = np.dot(h, self.W)  # Compute weighted input
        self.z = weighted_sum + self.b     # Add bias term
        self.activations = self.activation_function.forward(self.z)
        
        if self.is_training and self.dropout_rate > 0:
            self.dropout_mask = np.random.binomial(1, 1 - self.dropout_rate,
                                                    size=self.activations.shape)
            return self.activations * self.dropout_mask / (1-self.dropout_rate)
        
        return self.activations

    def backward(self, h: np.ndarray, delta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply backpropagation to this layer and return the weight and bias gradients

        :param h: input to this layer
        :param delta: delta term from layer above
        :return: (weight gradients, bias gradients)
        """
        # If dropout was applied, ensure that the gradient respects the same mask.
        if self.is_training and self.dropout_rate > 0 and self.dropout_mask is not None:
            delta = delta * self.dropout_mask / (1 - self.dropout_rate)
    
        dL_dW = np.dot(h.T, delta) #Compute weight gradient
        dL_db = np.sum(delta, axis=0, keepdims=True) #Compute bias gradient	
        self.delta = delta #Store delta	
        return dL_dW, dL_db

class MultilayerPerceptron:
    def __init__(self, layers: Tuple[Layer]):
        """
        Create a multilayer perceptron (densely connected multilayer neural network)
        :param layers: list or Tuple of layers
        """
        self.layers = layers

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        This takes the network input and computes the network output (forward propagation)
        :param x: network input
        :return: network output
        """
        current_output = x  # Start with the input data and stores the activations as they propagate through each layer.
        for layer in self.layers:
            current_output = layer.forward(current_output)
        return current_output 

    def backward(self, loss_grad: np.ndarray, input_data: np.ndarray) -> Tuple[list, list]:
        """
        Applies backpropagation to compute the gradients of the weights and biases for all layers in the network
        :param loss_grad: gradient of the loss function
        :param input_data: network's input data
        :return: (List of weight gradients for all layers, List of bias gradients for all layers)
        """
        layers = self.layers
        deltas = []

        output_layer = layers[-1]
        if isinstance(output_layer.activation_function, Softmax):
            delta = loss_grad
        else:
            delta = loss_grad * output_layer.activation_function.derivative(output_layer.z)
        deltas.insert(0, delta)

        for i in reversed(range(len(layers) - 1)):
            current_layer = layers[i]
            next_layer = layers[i + 1]
            delta_next = deltas[0]
            delta_current = (delta_next @ next_layer.W.T) * current_layer.activation_function.derivative(current_layer.z)
            deltas.insert(0, delta_current)

        dl_dw_all, dl_db_all = [], []
        for i, layer in enumerate(layers):
            h_prev = input_data if i == 0 else layers[i - 1].activations
            dL_dW, dL_db = layer.backward(h_prev, deltas[i])
            dl_dw_all.append(dL_dW)
            dl_db_all.append(dL_db)

        return dl_dw_all, dl_db_all
    
    def set_training_mode(self, is_training:bool):
        for layer in self.layers:
            layer.is_training = is_training
            
    def train(self, train_x: np.ndarray, train_y: np.ndarray, 
              val_x: np.ndarray, val_y: np.ndarray, 
              loss_func: LossFunction, learning_rate: float=1E-3, 
              batch_size: int=16, epochs: int=32, rmsprop: bool=False, beta: float=0.9, epsilon: float=1e-8, lr_decay: float=0.99, l2_lambda: float=0.01) -> Tuple[np.ndarray, np.ndarray]:
        """
        Train the multilayer perceptron

        :param train_x: full training set input of shape (n x d) n = number of samples, d = number of features
        :param train_y: full training set output of shape (n x q) n = number of samples, q = number of outputs per sample
        :param val_x: full validation set input
        :param val_y: full validation set output
        :param loss_func: instance of a LossFunction
        :param learning_rate: learning rate for parameter updates
        :param batch_size: size of each batch
        :param epochs: number of epochs
        :return:
        """
        print(learning_rate)
        print(batch_size)
        print(epochs)
        
        self.set_training_mode(True)
        training_losses = []
        validation_losses = []
        n_samples = train_x.shape[0]
        
        # Initialize RMSProp accumulators
        if rmsprop:
            rmsprop_cache_w = [np.zeros_like(layer.W) for layer in self.layers]
            rmsprop_cache_b = [np.zeros_like(layer.b) for layer in self.layers]
            
        best_val_loss = float('inf')
        patience = 20
        patience_counter = 0
        for epoch in range(epochs):
            # Shuffle data
            permutation = np.random.permutation(n_samples)
            train_x_shuffled, train_y_shuffled = train_x[permutation], train_y[permutation]
            epoch_loss = 0.0

            for batch_x, batch_y in batch_generator(train_x_shuffled, train_y_shuffled, batch_size):
                # Forward pass
                output = self.forward(batch_x)
                # Compute loss
                loss = np.mean(loss_func.loss(batch_y, output))
                epoch_loss += loss
                # Compute gradients
                loss_grad = loss_func.derivative(batch_y, output)
                grads_w, grads_b = self.backward(loss_grad, batch_x)
                for i, (layer, grad_w, grad_b) in enumerate(zip(self.layers, grads_w, grads_b)):
                    if rmsprop:
                        # Update RMSProp cache
                        rmsprop_cache_w[i] = beta * rmsprop_cache_w[i] + (1 - beta) * grad_w**2
                        rmsprop_cache_b[i] = beta * rmsprop_cache_b[i] + (1 - beta) * grad_b**2

                        # Update parameters using RMSProp
                        layer.W -= learning_rate * grad_w / (np.sqrt(rmsprop_cache_w[i]) + epsilon)
                        layer.b -= learning_rate * grad_b / (np.sqrt(rmsprop_cache_b[i]) + epsilon)
                    else:
                        # Vanilla SGD update
                        layer.W -= learning_rate * grad_w
                        layer.b -= learning_rate * grad_b
                    # Apply L2 regularization
                    layer.W -= learning_rate * l2_lambda * layer.W

            # Calculate epoch losses
            epoch_loss /= (n_samples / batch_size)
            training_losses.append(epoch_loss)
            # Validation loss
            self.set_training_mode(False)
            val_output = self.forward(val_x)
            val_loss = np.mean(loss_func.loss(val_y, val_output))
            validation_losses.append(val_loss)
            self.set_training_mode(True)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break
            
            learning_rate *= lr_decay
            
            # Print progress
            print(f"Epoch {epoch + 1}/{epochs} - "
                  f"Training Loss: {epoch_loss:.4f}, "
                  f"Validation Loss: {val_loss:.4f}")
            
        return np.array(training_losses), np.array(validation_losses)