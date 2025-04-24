import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

class BPNN:
    def __init__(self, layer_sizes, id, optimizer='adam', activation='tanh', output_activation=None):
        self.layer_sizes = layer_sizes
        print(f'Initializing network with layer sizes: {layer_sizes}')
        self.number_of_layers = len(layer_sizes)
        self.id = id
        self.optimizer = optimizer
        self.activation = activation
        self.output_activation = output_activation  # Specific activation for output layer

        self.cache = {}
        self.gradients = {}
        
        # Load parameters after setting class attributes
        self.load_parameters()

        if self.optimizer == 'adam':
            self.initialize_adam()

    def load_parameters(self):
        try:
            with open(f'./params/params_{self.id}.npz', 'rb') as f:
                loaded_params = np.load(f)
                num_weights = self.number_of_layers - 1
                self.weights = [loaded_params[f'arr_{i}'] for i in range(num_weights)]
                self.biases = [loaded_params[f'arr_{i+num_weights}'] for i in range(num_weights)]
                print(f'Parameters loaded from params_{self.id}.npz')
        except FileNotFoundError:
            print(f'params_{self.id}.npz not found. Using randomly initialized parameters.')
            self.param_generator(self.activation)
        except (EOFError, KeyError) as e:
            print(f'Error loading parameters: {e}. Using randomly initialized parameters.')
            self.param_generator(self.activation)

    def param_generator(self, activation):
        # He initialization for ReLU
        if activation == 'relu' or activation == 'leaky_relu':
            np.random.seed(42)  # Set fixed seed for reproducibility
            self.weights = [np.random.randn(self.layer_sizes[i-1], self.layer_sizes[i]) * np.sqrt(2 / self.layer_sizes[i-1]) for i in range(1, self.number_of_layers)]
        # Xavier initialization for tanh/sigmoid
        elif activation == 'tanh' or activation == 'sigmoid':
            np.random.seed(42)  # Set fixed seed for reproducibility
            self.weights = [np.random.randn(self.layer_sizes[i-1], self.layer_sizes[i]) * np.sqrt(1 / self.layer_sizes[i-1]) for i in range(1, self.number_of_layers)]
        # Standard initialization for linear
        else:
            np.random.seed(42)  # Set fixed seed for reproducibility
            self.weights = [np.random.randn(self.layer_sizes[i-1], self.layer_sizes[i]) * 0.01 for i in range(1, self.number_of_layers)]
        
        # Zero initialization for biases
        self.biases = [np.zeros((1, self.layer_sizes[i])) for i in range(1, self.number_of_layers)]

    def activation_set(self, z, activation, derivative=False):
        # Implementing numerical stability for activations
        if activation == 'tanh':
            return (1 - np.tanh(z)**2) if derivative else np.tanh(z)
        elif activation == 'relu':
            return np.where(z > 0, 1, 0) if derivative else np.maximum(0, z)
        elif activation == 'leaky_relu':
            alpha = 0.01
            return np.where(z > 0, 1, alpha) if derivative else np.where(z > 0, z, alpha * z)
        elif activation == 'sigmoid':
            sig = 1 / (1 + np.exp(-z))
            return sig * (1 - sig) if derivative else sig
        elif activation == 'linear':
            return np.ones_like(z) if derivative else z
        else:
            raise ValueError(f'Invalid activation function: {activation}')

    def forward_propagation(self, X, activation=None, output_activation=None):
        """
        Forward propagation supporting batch processing
        X: input data (batch_size, input_features)
        """
        if activation is None:
            activation = self.activation
        if output_activation is None:
            output_activation = self.output_activation if self.output_activation else 'linear'
            
        a = X
        self.cache = {'A0': a}
        
        # Hidden layers
        for i in range(self.number_of_layers-2):
            z = np.dot(a, self.weights[i]) + self.biases[i]
            a = self.activation_set(z, activation)
            self.cache[f'Z{i+1}'] = z
            self.cache[f'A{i+1}'] = a
            
        # Output layer
        z = np.dot(a, self.weights[-1]) + self.biases[-1]
        output = self.activation_set(z, output_activation)
        
        self.cache[f'Z{self.number_of_layers-1}'] = z
        self.cache[f'A{self.number_of_layers-1}'] = output
        return output

    def mse(self, y, y_pred):
        """Mean squared error loss function"""
        return np.mean((y - y_pred)**2)
    
    def mse_derivative(self, y, y_pred):
        """Derivative of MSE loss function"""
        return 2 * (y_pred - y) / y.shape[0]

    def backward_propagation(self, y, activation=None, output_activation=None):
        """
        Backward propagation supporting batch processing
        y: target values (batch_size, output_features)
        """
        if activation is None:
            activation = self.activation
        if output_activation is None:
            output_activation = self.output_activation if self.output_activation else 'linear'
            
        y_pred = self.cache[f'A{self.number_of_layers-1}']
        m = y.shape[0]  # Batch size
        
        # Output layer gradient
        if output_activation == 'linear':
            dz = self.mse_derivative(y, y_pred)
        else:
            dz = self.mse_derivative(y, y_pred) * self.activation_set(
                self.cache[f'Z{self.number_of_layers-1}'], output_activation, derivative=True)
        
        self.gradients[f'dW{self.number_of_layers-1}'] = (1/m) * np.dot(self.cache[f'A{self.number_of_layers-2}'].T, dz)
        self.gradients[f'db{self.number_of_layers-1}'] = (1/m) * np.sum(dz, axis=0, keepdims=True)
        
        # Hidden layers gradient
        for l in range(self.number_of_layers-2, 0, -1):
            dz = np.dot(dz, self.weights[l].T) * self.activation_set(
                self.cache[f'Z{l}'], activation, derivative=True)
            self.gradients[f'dW{l}'] = (1/m) * np.dot(self.cache[f'A{l-1}'].T, dz)
            self.gradients[f'db{l}'] = (1/m) * np.sum(dz, axis=0, keepdims=True)

    def initialize_adam(self):
        """Initialize Adam optimizer parameters"""
        self.m_w = [np.zeros_like(w) for w in self.weights]
        self.v_w = [np.zeros_like(w) for w in self.weights]
        self.m_b = [np.zeros_like(b) for b in self.biases]
        self.v_b = [np.zeros_like(b) for b in self.biases]
        self.t = 0

    def update_parameters(self, learning_rate=None, optimizer=None, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """Update parameters using specified optimizer"""
        if learning_rate is None:
            learning_rate = self.learning_rate
        if optimizer is None:
            optimizer = self.optimizer
            
        if optimizer == 'adam':
            if not hasattr(self, 'm_w'):
                self.initialize_adam()
                
            self.t += 1
            for l in range(self.number_of_layers-1):
                # Update weights
                self.m_w[l] = beta1 * self.m_w[l] + (1 - beta1) * self.gradients[f'dW{l+1}']
                self.v_w[l] = beta2 * self.v_w[l] + (1 - beta2) * np.square(self.gradients[f'dW{l+1}'])
                m_w_corrected = self.m_w[l] / (1 - beta1**self.t)
                v_w_corrected = self.v_w[l] / (1 - beta2**self.t)
                self.weights[l] -= learning_rate * m_w_corrected / (np.sqrt(v_w_corrected) + epsilon)
                
                # Update biases
                self.m_b[l] = beta1 * self.m_b[l] + (1 - beta1) * self.gradients[f'db{l+1}']
                self.v_b[l] = beta2 * self.v_b[l] + (1 - beta2) * np.square(self.gradients[f'db{l+1}'])
                m_b_corrected = self.m_b[l] / (1 - beta1**self.t)
                v_b_corrected = self.v_b[l] / (1 - beta2**self.t)
                self.biases[l] -= learning_rate * m_b_corrected / (np.sqrt(v_b_corrected) + epsilon)
        
        elif optimizer == 'sgd':
            for l in range(self.number_of_layers-1):
                self.weights[l] -= learning_rate * self.gradients[f'dW{l+1}']
                self.biases[l] -= learning_rate * self.gradients[f'db{l+1}']

    def train(self, X, y, X_val, y_val, epochs, batch_size=1, learning_rate = 1e-5, 
              patience=3, error_limit=1e-5, generate_new_params=True, shuffle = True):
        """
        Train the network with mini-batch gradient descent
        
        Parameters:
        X: training input data
        y: training target data
        X_val: validation input data
        y_val: validation target data
        epochs: maximum number of epochs to train
        batch_size: mini-batch size
        learning_rate: learning rate for optimizer
        patience: epochs to wait for improvement before stopping
        error_limit: stop training if error falls below this value
        generate_new_params: whether to reinitialize parameters
        """
        self.learning_rate = learning_rate
        
        if generate_new_params:
            self.weights, self.biases = [], []
            self.param_generator(self.activation)
            self.save_parameters()
            self.load_parameters()

        # Ensure data is in correct format
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        if len(X_val.shape) == 1:
            X_val = X_val.reshape(-1, 1)
        if len(y_val.shape) == 1:
            y_val = y_val.reshape(-1, 1)
            
        n_samples = X.shape[0]
        error_log, val_error_log = [], []
        patience_counter = patience
        best_val_error = float('inf')
        best_weights, best_biases = None, None

        progress_bar = tqdm(range(epochs), desc="Training Progress", unit="epoch")
        for epoch in progress_bar:
            
            #shuffle data
            if shuffle:
                idx = np.random.permutation(n_samples)
                X = X[idx]
                y = y[idx]

            # Mini-batch training
            for i in range(0, n_samples, batch_size):
                end = min(i + batch_size, n_samples)
                X_batch = X[i:end]
                y_batch = y[i:end]
                
                # Forward propagation
                y_pred_batch = self.forward_propagation(X_batch)
                
                # Backward propagation and parameter update
                self.backward_propagation(y_batch)
                self.update_parameters()
            
            # Evaluate on training set
            y_pred = self.forward_propagation(X)
            training_error = self.mse(y, y_pred)
            error_log.append(training_error)
            
            # Evaluate on validation set
            y_val_pred = self.forward_propagation(X_val)
            val_error = self.mse(y_val, y_val_pred)
            val_error_log.append(val_error)
            
            progress_bar.set_postfix({'loss': training_error, 'val_loss': val_error, 'patience': patience_counter})
            
            # Early stopping logic
            if val_error < best_val_error:
                best_val_error = val_error
                patience_counter = patience
                # Save best model
                best_weights = [w.copy() for w in self.weights]
                best_biases = [b.copy() for b in self.biases]
                self.save_parameters()
            else:
                patience_counter -= 1
                
            # Stopping criteria
            if patience_counter == 0 or training_error < error_limit:
                print(f'Epoch {epoch+1}/{epochs}, Training Loss: {training_error:.6f}, Validation Loss: {val_error:.6f}')
                print(f'Early stopping at epoch {epoch+1}. Best validation loss: {best_val_error:.6f}')
                
                # Restore best model if we stopped due to patience
                if best_weights is not None:
                    self.weights = best_weights
                    self.biases = best_biases
                    self.save_parameters()
                
                #plot loss
                self.plot_loss(error_log, val_error_log)
                    
                return {'loss': error_log, 'val_loss': val_error_log}

        #plot loss
        self.plot_loss(error_log, val_error_log)
                
        print(f'Completed {epochs} epochs. Training Loss: {training_error:.6f}, Validation Loss: {val_error:.6f}')
        return {'loss': error_log, 'val_loss': val_error_log}

    def save_parameters(self):
        """Save model parameters to file"""
        os.makedirs('./params', exist_ok=True)
        np.savez(f'./params/params_{self.id}.npz', *self.weights, *self.biases)

    def predict(self, X):
        """Make predictions for input data"""
        # Ensure data is in correct format
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        return self.forward_propagation(X)
    
    def plot_loss(self, error_log, val_error_log):
        """Plot training and validation loss"""
        plt.figure(figsize=(10, 5))
        plt.plot(error_log, label='Training Loss')
        plt.plot(val_error_log, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title(f'Training and Validation Loss for Network ID: {self.id}')
        plt.legend()
        plt.show()