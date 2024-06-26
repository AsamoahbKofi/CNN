import numpy as np
class Convlayer:
    def __init__(self, num_filters, filter_size, stride=1, padding=0):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        self.filters = np.random.randn(num_filters, filter_size, filter_size) / filter_size**2
        self.biases = np.zeros(num_filters)

    def activation(self,x):
        return np.maximum(0,x)
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    
    def local_response_normalization(self, x, alpha=1e-4, beta=0.75):
        normalized_output = (x / np.mean(x, axis=(1, 2, 3), keepdims=True)) / np.sqrt(np.sum(x**2, axis=(1, 2, 3),\
                     keepdims=True) + alpha) ** beta
        return normalized_output
    
    def apply_dropout(self, x, rate):
        keep_prob = 1 - rate
        mask = np.random.binomial(1, keep_prob, size=x.shape) / keep_prob
        return x * mask

    

    
    def max_pooling(self, x, pool_size=2):
        batch_size, output_height, output_width, num_filters = x.shape
        pooled_output = np.zeros((batch_size, output_height // pool_size, output_width // pool_size, num_filters))
        for i in range(0, output_height, pool_size):
            for j in range(0, output_width, pool_size):
                pooled_output[:, i // pool_size, j // pool_size, :] = np.max(x[:, i:i + pool_size, j:j + pool_size, :], axis=(1, 2))
        return pooled_output

            
    def flatten(self, x):
        return x.reshape(x.shape[0], -1)
    
    def fully_layer(self, x, num_neurons):
        self.weights = np.random.randn(x.shape[1], num_neurons) / np.sqrt(x.shape[1])
        self.biases = np.zeros(num_neurons)
        return np.dot(x, self.weights) + self.biases
    

        
    
   
    def forward_pass(self, input_image, num_neurons=1000):
        batch_size, image_height, image_width, num_channels = input_image.shape
        output_height = (image_height + 2 * self.padding - self.filter_size) // self.stride + 1
        output_width = (image_width + 2 * self.padding - self.filter_size) // self.stride + 1
        
        padded_image = np.pad(input_image, ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)), 'constant')
        output = np.zeros((batch_size, output_height, output_width, self.num_filters))
        
        # Convolution
        for i in range(0, output_height):
            for j in range(0, output_width):
                patch = padded_image[:, i * self.stride:i * self.stride + self.filter_size, j * self.stride:j * self.stride + self.filter_size, :]
                output[:, i, j, :] = np.sum(patch * self.filters, axis=(1, 2, 3)) + self.biases
        
        # Activation
        output = self.activation(output)
        
        # Local Response Normalization
        output = self.local_response_normalization(output)
    
    # Max Pooling
        output = self.max_pooling(output, pool_size=2)
        
        # Flatten
        self.flatten_output = self.flatten(output)
        
        # Fully Connected Layer
        output = self.fully_layer(self.flatten_output, num_neurons)
    
    # Softmax
        output = self.softmax(output)
        
        return output

# backward pass up next ----> stay tuned for this part as it involves backpropagation and gradient descent.

               
    
