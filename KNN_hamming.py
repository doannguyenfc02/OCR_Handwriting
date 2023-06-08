import csv
import numpy as np

# Đọc dữ liệu từ tệp train
train_data = []
with open('mnist_train.csv', 'r') as file:
    reader = csv.reader(file)
    next(reader)  # Bỏ qua dòng đầu tiên (tiêu đề cột)
    for row in reader:
        train_data.append(row)

# Đọc dữ liệu từ tệp test
test_data = []
with open('mnist_test2000.csv', 'r') as file:
    reader = csv.reader(file)
    next(reader)  # Bỏ qua dòng đầu tiên (tiêu đề cột)
    for row in reader:
        test_data.append(row)

# Chuyển đổi dữ liệu thành mảng numpy
train_features = np.array([list(map(int, sample[1:])) for sample in train_data])
train_labels = np.array([int(sample[0]) for sample in train_data])
test_features = np.array([list(map(int, sample[1:])) for sample in test_data])
test_labels = np.array([int(sample[0]) for sample in test_data])

# Hàm tính khoảng cách Hamming giữa hai vectơ
def hamming_distance(a, b):
    return np.count_nonzero(a != b)

# Hàm KNN với xử lý nhiễu
def KNN(k, train_features, train_labels, test_features, threshold):
    predicted_labels = []
    for test_sample in test_features:
        distances = [hamming_distance(test_sample, train_sample) for train_sample in train_features]  # Tính khoảng cách Hamming
        nearest_indices = np.argsort(distances)[:k]  # Chỉ mục của k vectơ đặc trưng gần nhất
        nearest_labels = train_labels[nearest_indices]  # Nhãn tương ứng với k vectơ đặc trưng gần nhất
        predicted_label = np.argmax(np.bincount(nearest_labels))  # Nhãn xuất hiện nhiều nhất trong k láng giềng
        predicted_labels.append(predicted_label)
    return predicted_labels

# Nhận dạng chữ số viết tay với xử lý nhiễu
k = 3  # Số láng giềng trong KNN
threshold = 40  # Ngưỡng để xử lý nhiễu
predicted_labels = KNN(k, train_features, train_labels, test_features, threshold)

# Đánh giá mô hình
accuracy = np.mean(predicted_labels == test_labels) * 100
print("Độ chính xác của mô hình: {:.2f}%".format(accuracy))
