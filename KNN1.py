import numpy as np
import pandas as pd

# Định nghĩa hàm tính khoảng cách Euclidean
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

# Định nghĩa hàm dự đoán nhãn cho tập kiểm tra
def predict(X_train, y_train, X_test, k):
    y_pred = []
    
    for test_point in X_test:
        distances = []
        
        # Tính khoảng cách từ điểm kiểm tra đến tất cả các điểm huấn luyện
        for train_point in X_train:
            distance = euclidean_distance(test_point, train_point)
            distances.append(distance)
        
        # Lấy chỉ mục của k láng giềng gần nhất
        k_indices = np.argsort(distances)[:k]
        
        # Lấy nhãn của các láng giềng gần nhất
        k_labels = y_train[k_indices]
        
        # Đếm số lượng xuất hiện của từng nhãn
        label_counts = np.bincount(k_labels)
        
        # Chọn nhãn xuất hiện nhiều nhất là nhãn dự đoán cho điểm dữ liệu kiểm tra
        y_pred.append(np.argmax(label_counts))
    
    return y_pred

# Đọc dữ liệu huấn luyện
train_data = pd.read_csv('mnist_train.csv')

# Chia tập huấn luyện thành các đặc trưng (X_train) và nhãn (y_train)
X_train = train_data.drop('label', axis=1).values
y_train = train_data['label'].values

# Đọc dữ liệu kiểm tra
test_data = pd.read_csv('mnist_test.csv')

# Chia tập kiểm tra thành các đặc trưng (X_test) và nhãn (y_test)
X_test = test_data.drop('label', axis=1).values
y_test = test_data['label'].values

# Cài đặt mô hình k-NN
k = 3
y_pred = predict(X_train, y_train, X_test, k)

# Đánh giá mô hình
accuracy = np.mean(y_pred == y_test)
print("Độ chính xác trên tập kiểm tra: ", accuracy)