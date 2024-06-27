import numpy as np

class conv2d:
    def __init__(self, weight, bias, num_filter, filter_size, neuron):
        self.weight = weight
        self.bias = bias
        self.num_filter = num_filter
        self.filter_size = filter_size
        self.neuron = neuron

    def relu_activation(self, x):
        return np.maximum(0, x)
    
    def local_response_normalization(self, x, alpha=1e-4, beta=0.75):
        normalized_output = (x / np.mean(x, axis=(1, 2, 3), keepdims=True)) / np.sqrt(np.sum(x**2, axis=(1, 2, 3), keepdims=True) + alpha) ** beta
        return normalized_output
    
    def max_pooling(self, pooling_size, output):
        batch_size, output_height, output_width, num_channels = output.shape
        max_pool_height = output_height // pooling_size
        max_pool_width = output_width // pooling_size

        max_pool_output = np.zeros((batch_size, max_pool_height, max_pool_width, num_channels))

        for b in range(batch_size):
            for c in range(num_channels):
                for i in range(max_pool_height):
                    for j in range(max_pool_width):
                        h_start = i * pooling_size
                        h_end = h_start + pooling_size
                        w_start = j * pooling_size
                        w_end = w_start + pooling_size
                        
                        # Extract the region for max pooling
                        pool_region = output[b, h_start:h_end, w_start:w_end, c]
                        # Assign the maximum value to the corresponding position in the output
                        max_pool_output[b, i, j, c] = np.max(pool_region)
                        
        return max_pool_output
    
    def flatten(self, x):
        return x.reshape(x.shape[0], -1)
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # Stability improvement
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def fully_connected_layer(self, x):
        return np.dot(x, self.weight) + self.bias

    def apply_dropout(self, x, rate):
        keep_prob = 1 - rate
        mask = np.random.binomial(1, keep_prob, size=x.shape) / keep_prob
        return x * mask

    def forward_pass(self, image, kernel, stride, padding):
        batch_size, image_height, image_width, num_channels = image.shape
        kernel_height, kernel_width, kernel_channels, num_filters = kernel.shape
        assert num_channels == kernel_channels, "Kernel channels must match image channels."
        
        padded_image = np.pad(image, ((0, 0), (padding, padding), (padding, padding), (0, 0)), mode="constant")
        padded_image_height, padded_image_width = padded_image.shape[1:3]

        output_height = (image_height + 2 * padding - kernel_height) // stride + 1
        output_width = (image_width + 2 * padding - kernel_width) // stride + 1

        output = np.zeros((batch_size, output_height, output_width, num_filters))

        for i in range(output_height):
            for j in range(output_width):
                h_start = i * stride
                h_end = h_start + kernel_height
                w_start = j * stride
                w_end = w_start + kernel_width

                region = padded_image[:, h_start:h_end, w_start:w_end, :]
                for f in range(num_filters):
                    output[:, i, j, f] = np.sum(region * kernel[:, :, :, f], axis=(1, 2, 3)) + self.bias[f]

        output = self.relu_activation(output)
        output = self.local_response_normalization(output)
        output = self.max_pooling(2, output)
        output = self.flatten(output)
        output = self.fully_connected_layer(output)
        output = self.softmax(output)

        return output


# Generate random data for testing
np.random.seed(0)
batch_size = 2
image_height = 8
image_width = 8
num_channels = 3
num_filters = 2
filter_size = 3
num_neurons = 10
stride = 1
padding = 1

# Create random input image
image = np.random.randn(batch_size, image_height, image_width, num_channels)
kernel = np.random.randn(filter_size, filter_size, num_channels, num_filters)
weight = np.random.randn((image_height // 2) * (image_width // 2) * num_filters, num_neurons)
bias = np.random.randn(num_neurons)

conv_layer = conv2d(weight=weight, bias=bias, num_filter=num_filters, filter_size=filter_size, neuron=num_neurons)
output = conv_layer.forward_pass(image, kernel, stride, padding)

# Print the output
print("Output of forward pass:\n", output)
print(output.shape)
