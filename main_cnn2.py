import numpy as np
import pandas as pd
import random
from PIL import Image


# 1. Chuẩn bị dữ liệu
data = pd.read_csv('mnist_train.csv')
X = data.iloc[:, 1:].values / 255.0
y = data.iloc[:, 0].values

# 2. Tiền xử lý dữ liệu
def train_test_split(X, y, test_size=0.2, random_state=None):
    if random_state:
        random.seed(random_state)
    indices = list(range(len(X)))
    random.shuffle(indices)
    split_index = int(len(X) * (1 - test_size))
    X_train = X[indices[:split_index]]
    y_train = y[indices[:split_index]]
    X_test = X[indices[split_index:]]
    y_test = y[indices[split_index:]]
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape và chuẩn hóa dữ liệu
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255.0
X_test /= 255.0

# Chuyển đổi nhãn sang dạng one-hot encoding
num_classes = 10

def to_categorical(labels, num_classes):
    one_hot_labels = np.zeros((labels.shape[0], num_classes))
    for i in range(labels.shape[0]):
        label = labels[i]
        one_hot_labels[i][label] = 1
    return one_hot_labels

y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# 3. Xây dựng mô hình CNN
def initialize_weights(shape):
    return np.random.randn(*shape) * 0.01

def initialize_biases(shape):
    return np.zeros(shape)

def convolution(input, kernel, bias, stride, padding):
    input_height, input_width, input_channels, _ = input.shape
    #input_height, input_width, input_channels = input.shape
    kernel_height, kernel_width, input_channels, num_filters = kernel.shape

    #kernel_height, kernel_width, _, num_filters = kernel.shape
    output_height = int((input_height - kernel_height + 2 * padding) / stride) + 1
    output_width = int((input_width - kernel_width + 2 * padding) / stride) + 1
    output = np.zeros((output_height, output_width, num_filters))

    padded_input = np.pad(input, ((padding, padding), (padding, padding), (0, 0)), mode='constant')

    for h in range(output_height):
        for w in range(output_width):
            for f in range(num_filters):
                vertical_start = h * stride
                vertical_end = vertical_start + kernel_height
                horizontal_start = w * stride
                horizontal_end = horizontal_start + kernel_width

                receptive_field = padded_input[vertical_start:vertical_end, horizontal_start:horizontal_end, :]
                convolution_result = np.sum(receptive_field * kernel[:, :, :, f]) + bias[f]
                output[h, w, f] = convolution_result

    return output
def relu(x):
    return np.maximum(x, 0)

def max_pooling(image, size=2, stride=2):
    height, width, depth = image.shape

    output_height = int((height - size) / stride) + 1
    output_width = int((width - size) / stride) + 1
    output_depth = depth

    output = np.zeros((output_height, output_width, output_depth))

    for h in range(output_height):
        for w in range(output_width):
            h_start = h * stride
            w_start = w * stride
            h_end = h_start + size
            w_end = w_start + size

            image_patch = image[h_start:h_end, w_start:w_end, :]
            output[h, w] = np.amax(image_patch, axis=(0, 1))

    return output

def flatten(x):
    return x.reshape(-1)

def dense(x, weights, biases):
    return np.dot(x, weights) + biases

def softmax(x):
    exps = np.exp(x)
    return exps / np.sum(exps)

# Khởi tạo mô hình
model = [
    {'type': 'conv', 'filters': 32, 'kernel_size': (3, 3), 'stride': 1, 'padding': 0},
    {'type': 'activation', 'activation': relu},
    {'type': 'pool', 'size': 2, 'stride': 2},
    {'type': 'flatten'},
    {'type': 'dense', 'units': 128, 'activation': relu},
    {'type': 'dense', 'units': 10, 'activation': softmax}
]

# Khởi tạo trọng số và bias
# Khởi tạo trọng số và bias
weights = []
biases = []
for layer in model:
    if layer['type'] == 'conv':
        filters = layer['filters']
        kernel_size = layer['kernel_size']
        depth = X_train.shape[3]
        weights.append(initialize_weights((filters, kernel_size[0], kernel_size[1], depth, filters)))
        biases.append(initialize_biases((filters,)))
    elif layer['type'] == 'dense':
        units = layer['units']
        input_dim = X_train.shape[1] * X_train.shape[2] * X_train.shape[3]
        weights.append(initialize_weights((input_dim, units)))
        biases.append(initialize_biases((units,)))
# Huấn luyện mô hình
learning_rate = 0.001
batch_size = 128
num_epochs = 10

num_samples = X_train.shape[0]
num_batches = int(np.ceil(num_samples / batch_size))

for epoch in range(num_epochs):
    for batch in range(num_batches):
        start_idx = batch * batch_size
        end_idx = min((batch + 1) * batch_size, num_samples)
        batch_images = X_train[start_idx:end_idx]
        batch_labels = y_train[start_idx:end_idx]

        # Feedforward
        output = batch_images
        for i, layer in enumerate(model):
            if layer['type'] == 'conv':
                filters = layer['filters']
                kernel = weights[i]
                bias = biases[i]
                stride = layer['stride']
                padding = layer['padding']
                #kernel_height, kernel_width, input_channels, num_filters = kernel.shape
                #kernel_height, kernel_width, input_channels = kernel.shape
                kernel_height, kernel_width, input_channels = kernel.shape[:3]
                output = convolution(output, kernel, bias, stride, padding)
                output = layer['activation'](output)
            elif layer['type'] == 'pool':
                size = layer['size']
                #stride = layer['stride
                stride = layer['stride']
                output = max_pooling(output, size, stride)
            elif layer['type'] == 'flatten':
                output = flatten(output)
            elif layer['type'] == 'dense':
                weights = weights[i]
                biases = biases[i]
                output = dense(output, weights, biases)
                output = layer['activation'](output)
        
        # Tính toán loss và gradient
        loss = -np.sum(batch_labels * np.log(output))
        gradient = output - batch_labels
        
        # Backpropagation
        for i in range(len(model) - 1, -1, -1):
            layer = model[i]
            if layer['type'] == 'conv':
                gradient = gradient * layer['activation'](output, derivative=True)
                kernel = weights[i]
                bias = biases[i]
                stride = layer['stride']
                padding = layer['padding']
                padded_output = np.pad(output, ((padding, padding), (padding, padding), (0, 0)), mode='constant')
                kernel_gradient = convolution(padded_output, gradient, stride=stride)

                weights[i] -= learning_rate * kernel_gradient
                biases[i] -= learning_rate * np.sum(gradient, axis=(0, 1))

                gradient = convolution(gradient, np.rot90(kernel, 2), stride=1, padding=0)

            elif layer['type'] == 'dense':
                gradient = gradient * layer['activation'](output, derivative=True)
                layer_weights = weights[i]  # Rename the variable to avoid overriding weights
                layer_biases = biases[i]  # Rename the variable to avoid overriding biases
                weights_gradient = np.dot(output.T, gradient)
                biases_gradient = np.sum(gradient, axis=0)

                layer_weights -= learning_rate * weights_gradient  # Use the renamed variable
                layer_biases -= learning_rate * biases_gradient  # Use the renamed variable

                gradient = np.dot(gradient, layer_weights.T)  # Use the renamed variable

                
        if batch % 100 == 0:
            print(f"Epoch {epoch+1}/{num_epochs} - Batch {batch+1}/{num_batches} - Loss: {loss:.4f}")
# 5. Đánh giá mô hình
def predict(image):
    output = image
    for i, layer in enumerate(model):
        if layer['type'] == 'conv':
            kernel = weights[i]
            bias = biases[i]
            stride = layer['stride']
            padding = layer['padding']
            output = convolution(output, kernel, bias, stride, padding)
            output = layer['activation'](output)
        elif layer['type'] == 'pool':
            size = layer['size']
            stride = layer['stride']
            output = max_pooling(output, size, stride)
        elif layer['type'] == 'flatten':
            output = flatten(output)
        elif layer['type'] == 'dense':
            weights = weights[i]
            biases = biases[i]
            output = dense(output, weights, biases)
            output = layer['activation'](output)
    return output

def evaluate(X, y):
    num_samples = X.shape[0]
    num_correct = 0
    
    for i in range(num_samples):
        image = X[i]
        label = np.argmax(y[i])
        
        prediction = predict(image)
        predicted_label = np.argmax(prediction)
        
        if predicted_label == label:
            num_correct += 1

    accuracy = num_correct / num_samples
    return accuracy

accuracy = evaluate(X_test, y_test)
print("Accuracy:", accuracy)
# 6. Nhận dạng ảnh test
def preprocess_image(image_path):
    image = Image.open(image_path).convert('L')
    image = image.resize((28, 28))
    image_array = np.array(image) / 255.0
    image_array = image_array.reshape(1, 28, 28, 1)
    return image_array

def predict_image(image_path):
    image_array = preprocess_image(image_path)
    prediction = predict(image_array)
    predicted_label = np.argmax(prediction)
    return predicted_label

image_path = 'test.png'
predicted_label = predict_image(image_path)
print("Predicted label:", predicted_label)




