import pandas as pd
import numpy as np

# Đọc dữ liệu từ file CSV
data = pd.read_csv('letter.csv')

# Chia dữ liệu thành features (X) và labels (y)
X = data.iloc[:, 1:].values
y = data.iloc[:, 0].values

# Số lượng mẫu trong tập kiểm tra (20%)
test_size = int(len(X) * 0.2)

# Tạo ngẫu nhiên chỉ số của các mẫu trong tập kiểm tra
np.random.seed(42)
test_indices = np.random.choice(len(X), test_size, replace=False)

# Tạo tập huấn luyện và tập kiểm tra dựa trên chỉ số đã chọn
X_train = np.delete(X, test_indices, axis=0)
y_train = np.delete(y, test_indices)
X_test = X[test_indices]
y_test = y[test_indices]

# Hàm tính khoảng cách Euclid
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

# Hàm dự đoán nhãn của một mẫu dựa trên K láng giềng gần nhất
def predict_knn(X_train, y_train, x_test, k):
    distances = []
    for i in range(len(X_train)):
        dist = euclidean_distance(X_train[i], x_test)
        distances.append((dist, y_train[i]))
    distances.sort(key=lambda x: x[0])  # Sắp xếp theo khoảng cách tăng dần
    neighbors = distances[:k]  # Lấy K láng giềng gần nhất
    labels = [neighbor[1] for neighbor in neighbors]
    predicted_label = max(set(labels), key=labels.count)  # Nhãn xuất hiện nhiều nhất
    return predicted_label

# Hàm dự đoán nhãn cho tập kiểm tra
def predict(X_train, y_train, X_test, k):
    predictions = []
    for i in range(len(X_test)):
        prediction = predict_knn(X_train, y_train, X_test[i], k)
        predictions.append(prediction)
    return predictions

# Khởi tạo số láng giềng gần nhất (K)
k = 5

# Dự đoán nhãn cho tập kiểm tra
y_pred = predict(X_train, y_train, X_test, k)

# Đánh giá độ chính xác
accuracy = np.sum(y_pred == y_test) / len(y_test)
print("Độ chính xác của mô hình kNN trên tập kiểm tra:", accuracy)
