import numpy as np
from PIL import Image

# Hàm kích hoạt ReLU
def relu(x):
    return np.maximum(0, x)

# Hàm kích hoạt softmax
def softmax(x):
    exps = np.exp(x - np.max(x))
    return exps / np.sum(exps)

# Hàm huấn luyện mô hình
def train(X, y, X_test, y_test, num_epochs, learning_rate, hidden_size):
    np.random.seed(0)
    
    # Khởi tạo trọng số ngẫu nhiên
    W1 = np.random.randn(X.shape[1], hidden_size) * 0.01
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.randn(hidden_size, num_classes) * 0.01
    b2 = np.zeros((1, num_classes))
    
    m = X.shape[0]  # Số lượng mẫu huấn luyện
    
    for epoch in range(num_epochs):
        # Forward pass
        Z1 = np.dot(X, W1) + b1
        A1 = relu(Z1)
        Z2 = np.dot(A1, W2) + b2
        A2 = softmax(Z2)
        
        # Backward pass
        dZ2 = A2 - y
        dW2 = np.dot(A1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m
        dA1 = np.dot(dZ2, W2.T)
        dZ1 = np.multiply(dA1, np.int64(A1 > 0))
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m
        
        # Cập nhật trọng số
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        
        # Đánh giá độ chính xác trên tập huấn luyện sau mỗi epoch
        if epoch % 100 == 0:
            # Tính độ chính xác trên tập huấn luyện
            train_Z1 = np.dot(X, W1) + b1
            train_A1 = relu(train_Z1)
            train_Z2 = np.dot(train_A1, W2) + b2
            train_A2 = softmax(train_Z2)
            train_acc = np.mean(np.argmax(train_A2, axis=1) == np.argmax(y, axis=1))
            
            # Tính độ chính xác trên tập kiểm tra
            test_Z1 = np.dot(X_test, W1) + b1
            test_A1 = relu(test_Z1)
            test_Z2 = np.dot(test_A1, W2) + b2
            test_A2 = softmax(test_Z2)
            test_acc = np.mean(np.argmax(test_A2, axis=1) == np.argmax(y_test, axis=1))
            
            print(f"Epoch {epoch}: Training Accuracy = {train_acc}, Test Accuracy = {test_acc}")
    
    return W1, b1, W2, b2

# Đọc dữ liệu từ tập mnist_train.csv
train_data = np.genfromtxt('mnist_train.csv', delimiter=',', skip_header=1)
X_train = train_data[:, 1:] / 255.0  # Chuẩn hóa giá trị pixel về khoảng [0, 1]
y_train = train_data[:, 0].astype(int)  # Nhãn lớp
# Chuyển đổi nhãn thành one-hot encoding
num_classes = 10
y_one_hot = np.zeros((y_train.shape[0], num_classes))
for i in range(y_train.shape[0]):
    y_one_hot[i, y_train[i]] = 1

# Đọc dữ liệu từ tập mnist_test.csv
test_data = np.genfromtxt('mnist_test.csv', delimiter=',', skip_header=1)
X_test = test_data[:, 1:] / 255.0  # Chuẩn hóa giá trị pixel về khoảng [0, 1]
y_test = test_data[:, 0].astype(int)  # Nhãn lớp
# Chuyển đổi nhãn thành one-hot encoding
y_test_one_hot = np.zeros((y_test.shape[0], num_classes))
for i in range(y_test.shape[0]):
    y_test_one_hot[i, y_test[i]] = 1

# Cài đặt các tham số huấn luyện
num_epochs = 10
learning_rate = 0.001
hidden_size = 128

# Huấn luyện mô hình
W1, b1, W2, b2 = train(X_train, y_one_hot, X_test, y_test, num_epochs, learning_rate, hidden_size)

# Đọc ảnh test.png
image = Image.open('test2.png').convert('L')
image_array = np.array(image) / 255.0
image_array = image_array.reshape(1, -1)

# Forward pass để dự đoán nhãn của ảnh
Z1 = np.dot(image_array, W1) + b1
A1 = relu(Z1)
Z2 = np.dot(A1, W2) + b2
A2 = softmax(Z2)

# Nhãn dự đoán là chỉ mục có xác suất cao nhất
predicted_label = np.argmax(A2)

print("Predicted label:", predicted_label)

