import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image


# 1. Chuẩn bị dữ liệu
data = pd.read_csv('mnist_train.csv')
X = data.iloc[:, 1:].values / 255.0
y = data.iloc[:, 0].values
X = X.reshape(-1, 28, 28, 1)
y = np.eye(10)[y]  # One-hot encoding

# 2. Tiền xử lý dữ liệu
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# 3. Xây dựng mô hình CNN
class Conv2D:
    def __init__(self, num_filters, kernel_size, input_shape):
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.input_shape = input_shape
        self.filters = np.random.randn(kernel_size, kernel_size, input_shape[3], num_filters) / np.sqrt(kernel_size * kernel_size * input_shape[3])
    def iterate_regions(self, image):
        h, w = image.shape[1], image.shape[2]
        kh, kw = self.kernel_size, self.kernel_size

        for i in range(h - kh + 1):
            for j in range(w - kw + 1):
                im_region = image[:, i:i + kh, j:j + kw, :]
                yield im_region, i, j

    def forward(self, input):
        self.last_input = input
        h, w, _, _ = input.shape
        num_filters = self.num_filters
        output = np.zeros((input.shape[0], h - self.kernel_size + 1, w - self.kernel_size + 1, num_filters))

        for im_region, i, j in self.iterate_regions(input):
            output[:, i, j] = np.sum(im_region * self.filters, axis=(1, 2, 3))

        return output


class MaxPooling2D:
    def __init__(self, pool_size):
        self.pool_size = pool_size

    def iterate_regions(self, image):
        h, w, _ = image.shape
        ph, pw = self.pool_size, self.pool_size

        for i in range(h // ph):
            for j in range(w // pw):
                im_region = image[i * ph:(i + 1) * ph, j * pw:(j + 1) * pw]
                yield im_region, i, j

    def forward(self, input):
        self.last_input = input
        h, w, num_filters = input.shape[1], input.shape[2], input.shape[3]
        output = np.zeros((input.shape[0], h // self.pool_size, w // self.pool_size, num_filters))

        for im_region, i, j in self.iterate_regions(input):
            output[:, i, j] = np.amax(im_region, axis=(1, 2))

        return output


class Flatten:
    def forward(self, input):
        self.last_input_shape = input.shape
        return input.reshape(input.shape[0], -1)


class Dense:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.randn(input_size, output_size) / np.sqrt(input_size)
        self.biases = np.zeros(output_size)

    def forward(self, input):
        self.last_input = input
        return np.dot(input, self.weights) + self.biases

    def backward(self, gradient):
        grad_input = np.dot(gradient, self.weights.T)
        grad_weights = np.dot(self.last_input.T, gradient)
        grad_biases = np.sum(gradient, axis=0)
        self.grad_weights = grad_weights  # Lưu gradient của trọng số
        self.grad_biases = grad_biases  # Lưu gradient của bias
        return grad_input

    def update_parameters(self, learning_rate):
        self.weights -= learning_rate * self.grad_weights
        self.biases -= learning_rate * self.grad_biases

# 4. Huấn luyện mô hình
class CNNModel:
    def __init__(self):
        self.conv1 = Conv2D(num_filters=32, kernel_size=3, input_shape=(28, 28, X.shape[3]))
        #self.conv1 = Conv2D(num_filters=32, kernel_size=3, input_shape=(28, 28, 1))
        self.pool1 = MaxPooling2D(pool_size=2)
        self.flatten = Flatten()
        self.dense = Dense(input_size=6272, output_size=10)  # Update input_size to match the output shape of flatten layer

    def forward(self, input):
        x = self.conv1.forward(input)
        x = self.pool1.forward(x)
        x = self.flatten.forward(x)
        x = self.dense.forward(x)
        return x

    def train(self, X_train, y_train, X_test, y_test, num_epochs, learning_rate, batch_size):
        num_batches = X_train.shape[0] // batch_size

        for epoch in range(1, num_epochs + 1):
            # Training
            for batch in range(num_batches):
                batch_X = X_train[batch * batch_size: (batch + 1) * batch_size]
                batch_y = y_train[batch * batch_size: (batch + 1) * batch_size]

                # Forward pass
                output = self.forward(batch_X)

                # Backward pass
                error = output - batch_y
                grad_output = error / batch_size
                grad_dense = self.dense.backward(grad_output)
                grad_flatten = self.flatten.backward(grad_dense)
                grad_pool = self.pool1.backward(grad_flatten)
                grad_conv = self.conv1.backward(grad_pool)

                # Update parameters
                self.dense.update_parameters(learning_rate)
                self.conv1.update_parameters(learning_rate)

            # Evaluation
            test_output = self.forward(X_test)
            accuracy = np.mean(np.argmax(test_output, axis=1) == np.argmax(y_test, axis=1))
            print(f"Epoch {epoch}: Accuracy = {accuracy * 100:.2f}%")

# Tạo mô hình CNN
model = CNNModel()

# Huấn luyện mô hình
num_epochs = 5
learning_rate = 0.001
batch_size = 128
model.train(X_train, y_train, X_test, y_test, num_epochs, learning_rate, batch_size)

# Đánh giá mô hình trên tập test
test_output = model.forward(X_test)
accuracy = np.mean(np.argmax(test_output, axis=1) == np.argmax(y_test, axis=1))
print(f"Accuracy: {accuracy * 100:.2f}%")

# Load ảnh test và tiền xử lý
image_path = 'test5.png'
image = Image.open(image_path).convert('L')
image = image.resize((28, 28))
image_array = np.array(image) / 255.0
image_array = image_array.reshape(1, 28, 28, 1)

# Sử dụng mô hình để dự đoán
prediction = model.forward(image_array)
predicted_label = np.argmax(prediction)

print("Predicted label:", predicted_label)
