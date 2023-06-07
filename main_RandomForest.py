#1. Chuẩn bị dữ liệu:
#Đọc dữ liệu từ tập tin CSV bằng thư viện pandas.

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


#2.Tiền xử lý dữ liệu:
#Tiêu chuẩn hóa đặc trưng bằng cách chia mỗi giá trị pixel cho 255 để nằm trong khoảng từ 0 đến 1.




# Tiêu chuẩn hóa đặc trưng
X_train = X_train / 255.0
X_test = X_test / 255.0

#3. Xây dựng và huấn luyện mô hình Random Forest:
#Sử dụng thư viện scikit-learn để xây dựng và huấn luyện mô hình Random Forest.

from sklearn.ensemble import RandomForestClassifier

# Tạo một đối tượng Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Huấn luyện mô hình trên tập huấn luyện
rf_model.fit(X_train, y_train)


#4. Đánh giá mô hình:
#Đánh giá mô hình trên tập kiểm tra bằng cách tính toán độ chính xác.
# Đánh giá mô hình trên tập kiểm tra
accuracy = rf_model.score(X_test, y_test)
print("Độ chính xác trên tập kiểm tra: ", accuracy)
