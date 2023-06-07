import numpy as np
import pandas as pd

# Đọc dữ liệu từ file CSV
df_train = pd.read_csv('mnist_train.csv')
df_test = pd.read_csv('mnist_test.csv')

# Chuyển đổi dữ liệu thành ma trận numpy
X_train = df_train.iloc[:, 1:].values
y_train = df_train.iloc[:, 0].values
X_test = df_test.iloc[:, 1:].values
y_test = df_test.iloc[:, 0].values

class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.num_classes = len(np.unique(y))
        self.tree = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        num_samples, num_features = X.shape
        unique_classes, counts = np.unique(y, return_counts=True)

        # Trường hợp dừng (đạt tới độ sâu tối đa hoặc không còn phân loại sai)
        if (self.max_depth is not None and depth >= self.max_depth) or len(unique_classes) == 1:
            # Trả về nút lá với nhãn phổ biến nhất
            return {'class': unique_classes[np.argmax(counts)]}

        # Tìm đặc trưng và ngưỡng tốt nhất để phân chia dữ liệu
        best_feature, best_threshold = None, None
        best_gini = 1.0
        for feature in range(num_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                gini = self._gini_index(X, y, feature, threshold)
                if gini < best_gini:
                    best_feature = feature
                    best_threshold = threshold
                    best_gini = gini

        # Tạo nút quyết định mới dựa trên đặc trưng và ngưỡng tốt nhất
        left_indices = X[:, best_feature] < best_threshold
        right_indices = X[:, best_feature] >= best_threshold
        left_tree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_tree = self._build_tree(X[right_indices], y[right_indices], depth + 1)

        return {'feature': best_feature, 'threshold': best_threshold, 'left': left_tree, 'right': right_tree}

    def _gini_index(self, X, y, feature, threshold):
        left_indices = X[:, feature] < threshold
        right_indices = X[:, feature] >= threshold
        left_classes, left_counts = np.unique(y[left_indices], return_counts=True)
        right_classes, right_counts = np.unique(y[right_indices], return_counts=True)
        left_gini = 1.0 - np.sum((left_counts / np.sum(left_counts)) ** 2)
        right_gini = 1.0 - np.sum((right_counts / np.sum(right_counts)) ** 2)

        num_left = len(y[left_indices])
        num_right = len(y[right_indices])
        total_samples = num_left + num_right

        gini_index = (num_left / total_samples) * left_gini + (num_right / total_samples) * right_gini

        return gini_index

    def predict(self, X):
        predictions = []
        for sample in X:
            prediction = self._predict_sample(sample, self.tree)
            predictions.append(prediction)
        return np.array(predictions)

    def _predict_sample(self, sample, node):
        if 'class' in node:
            return node['class']
        if sample[node['feature']] < node['threshold']:
            return self._predict_sample(sample, node['left'])
        else:
            return self._predict_sample(sample, node['right'])

# Xây dựng cây quyết định và huấn luyện trên tập huấn luyện
tree = DecisionTree(max_depth=10)
tree.fit(X_train, y_train)

# Dự đoán nhãn trên tập kiểm tra
y_pred = tree.predict(X_test)

# Đánh giá độ chính xác của cây quyết định
accuracy = np.sum(y_pred == y_test) / len(y_test)
print("Accuracy:", accuracy)
