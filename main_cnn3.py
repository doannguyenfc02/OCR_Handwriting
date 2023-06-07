import numpy as np
import pandas as pd
from PIL import Image

# Hàm train_test_split
def train_test_split(X, y, test_size=0.2, random_state=None):
    np.random.seed(random_state)

    num_samples = X.shape[0]
    indices = np.random.permutation(num_samples)

    num_test_samples = int(test_size * num_samples)

    X_test = X[indices[:num_test_samples]]
    y_test = y[indices[:num_test_samples]]
    X_train = X[indices[num_test_samples:]]
    y_train = y[indices[num_test_samples:]]

    return X_train, X_test, y_train, y_test

# Hàm activation ReLU
def relu(x):
    return np.maximum(0, x)

# Hàm tích chập
def convolution(X, kernel, bias):
    num_samples, input_height, input_width, input_channels = X.shape
    kernel_height, kernel_width, _, num_filters = kernel.shape
    output_height = input_height - kernel_height + 1
    output_width = input_width - kernel_width + 1

    conv_out = np.zeros((num_samples, output_height, output_width, num_filters))

    for n in range(num_samples):
        for k in range(num_filters):
            for i in range(output_height):
                for j in range(output_width):
                    conv_out[n, i, j, k] = np.sum(X[n, i:i+kernel_height, j:j+kernel_width, :] * kernel[:, :, :, k]) + bias[k]

    return conv_out

# Hàm flatten
def flatten(X):
    return X.reshape(X.shape[0], -1)

# Hàm fully connected layer
def fully_connected(X, weights, bias):
    weights_reshaped = weights.T  # Chuyển vị ma trận weights
    return np.dot(X, weights_reshaped) + bias

# Hàm softmax
def softmax(X):
    exps = np.exp(X - np.max(X, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)

# Chuẩn bị dữ liệu
data = pd.read_csv('mnist_train.csv')
X = data.iloc[:, 1:].values / 255.0
y = data.iloc[:, 0].values

X = X.reshape(-1, 28, 28, 1)
y_one_hot = np.zeros((y.shape[0], 10))
y_one_hot[np.arange(y.shape[0]), y] = 1

# Chia thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)

# Xây dựng mô hình CNN
kernel_size = (3, 3)
num_filters = 32
num_classes = 10

# Khởi tạo trọng số ngẫu nhiên
np.random.seed(0)
W1 = np.random.randn(*kernel_size, 1, num_filters)
b1 = np.zeros(num_filters)
W2 = np.random.randn(7 * 7 * num_filters, 128)
b2 = np.zeros(128)
W3 = np.random.randn(128, num_classes)
b3 = np.zeros(num_classes)


# Hàm huấn luyện mô hình
def train(X, y, num_epochs, learning_rate):
    global W1, b1, W2, b2, W3, b3

    for epoch in range(num_epochs):
        # Feedforward
        conv_out = convolution(X, W1, b1)
        conv_out_relu = relu(conv_out)
        flattened = flatten(conv_out_relu)
        fc_out = fully_connected(flattened, W2, b2)
        fc_out_relu = relu(fc_out)
        scores = fully_connected(fc_out_relu, W3, b3)
        probabilities = softmax(scores)

        # Tính loss
        num_samples = X.shape[0]
        loss = -np.sum(y * np.log(probabilities)) / num_samples

        # Gradient descent
        dscores = probabilities - y
        dW3 = np.dot(fc_out_relu.T, dscores) / num_samples
        db3 = np.sum(dscores, axis=0) / num_samples
        dfc_out_relu = np.dot(dscores, W3.T)
        dfc_out = dfc_out_relu.copy()
        dfc_out[fc_out <= 0] = 0
        dW2 = np.dot(flattened.T, dfc_out) / num_samples
        db2 = np.sum(dfc_out, axis=0) / num_samples
        dflattened = np.dot(dfc_out, W2.T)
        dconv_out_relu = dflattened.reshape(conv_out_relu.shape)
        dconv_out = dconv_out_relu.copy()
        dconv_out[conv_out <= 0] = 0
        dW1 = convolution(X.transpose(0, 3, 1, 2), dconv_out.transpose(3, 1, 2, 0), np.zeros_like(b1))
        dW1 = dW1.transpose(1, 2, 3, 0) / num_samples
        db1 = np.sum(dconv_out, axis=(0, 1, 2)) / num_samples

        # Cập nhật trọng số
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2
        W3 -= learning_rate * dW3
        b3 -= learning_rate * db3

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {loss}")

# Huấn luyện mô hình
num_epochs = 100
learning_rate = 0.001
train(X_train, y_train, num_epochs, learning_rate)
'''
# Đánh giá mô hình trên tập kiểm tra
conv_out_test = convolution(X_test, W1, b1)
conv_out_relu_test = relu(conv_out_test)
flattened_test = flatten(conv_out_relu_test)
fc_out_test = fully_connected(flattened_test, W2, b2)
fc_out_relu_test = relu(fc_out_test)
scores_test = fully_connected(fc_out_relu_test, W3, b3)
probabilities_test = softmax(scores_test)
predictions_test = np.argmax(probabilities_test, axis=1)
accuracy = np.mean(predictions_test == np.argmax(y_test, axis=1))
print(f"Accuracy on test set: {accuracy}")

'''
# Lớp dự đoán
def predict(image_path):
    image = Image.open(image_path).convert('L')
    image = image.resize((28, 28))
    image_data = np.array(image) / 255.0
    image_data = image_data.reshape(1, 28, 28, 1)

    conv_out = convolution(image_data, W1, b1)
    conv_out_relu = relu(conv_out)
    flattened = flatten(conv_out_relu)
    fc_out = fully_connected(flattened, W2, b2)
    fc_out_relu = relu(fc_out)
    scores = fully_connected(fc_out_relu, W3, b3)
    probabilities = softmax(scores)
    prediction = np.argmax(probabilities)

    return prediction

# Dự đoán một hình ảnh
image_path = 'test.png'
prediction = predict(image_path)
print(f"Predicted digit: {prediction}")