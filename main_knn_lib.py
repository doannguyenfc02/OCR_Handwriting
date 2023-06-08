from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

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

# Xây dựng mô hình KNN với số láng giềng là 5
knn_model = KNeighborsClassifier(n_neighbors=3)

# Huấn luyện mô hình trên tập huấn luyện
knn_model.fit(X_train, y_train)

# Đánh giá mô hình trên tập kiểm tra
accuracy = knn_model.score(X_test, y_test)
print("Độ chính xác trên tập kiểm tra: ", accuracy)
