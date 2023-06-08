import csv
import numpy as np

# Đọc dữ liệu từ tệp train
train_data = []
with open('mnist_train.csv', 'r') as file:
    reader = csv.reader(file)
    next(reader)  # Bỏ qua dòng đầu tiên (chứa tên cột)
    for row in reader:
        train_data.append(row)

# Đọc dữ liệu từ tệp test
test_data = []
with open('mnist_test4.csv', 'r') as file:
    reader = csv.reader(file)
    next(reader)  # Bỏ qua dòng đầu tiên (chứa tên cột)
    for row in reader:
        test_data.append(row)

# Chuyển đổi dữ liệu thành mảng numpy
train_features = np.array([list(map(int, sample[1:])) for sample in train_data])
train_labels = np.array([int(sample[0]) for sample in train_data])
test_features = np.array([list(map(int, sample[1:])) for sample in test_data])
test_labels = np.array([int(sample[0]) for sample in test_data])

# Hàm tính khoảng cách cosine
def cosine_distance(x1, x2):
    dot_product = np.dot(x1, x2)
    norm_x1 = np.linalg.norm(x1)
    norm_x2 = np.linalg.norm(x2)
    similarity = dot_product / (norm_x1 * norm_x2)
    distance = 1 - similarity
    return distance

# Hàm KNN
def KNN(train_features, train_labels, test_feature, k):
    distances = [cosine_distance(test_feature, train_feature) for train_feature in train_features]
    sorted_indices = np.argsort(distances)
    k_nearest_labels = train_labels[sorted_indices[:k]]
    predicted_label = np.argmax(np.bincount(k_nearest_labels))
    return predicted_label

# Lặp qua từng mẫu dữ liệu trong tập test và dự đoán nhãn
correct_predictions = 0
for i in range(len(test_features)):
    test_feature = test_features[i]
    test_label = test_labels[i]
    predicted_label = KNN(train_features, train_labels, test_feature, k=5)
    if predicted_label == test_label:
        correct_predictions += 1

accuracy = correct_predictions / len(test_features)
print("Accuracy: {:.2f}%".format(accuracy * 100))
