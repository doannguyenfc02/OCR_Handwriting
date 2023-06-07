import pandas as pd
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Đọc dữ liệu từ file CSV
data = pd.read_csv('letter.csv')

# Chia dữ liệu thành features (X) và labels (y)
X = data.iloc[:, 1:].values
y = data.iloc[:, 0].values

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Khởi tạo mô hình kNN
k = 5  # Số láng giềng gần nhất
knn = KNeighborsClassifier(n_neighbors=k)

# Huấn luyện mô hình
knn.fit(X_train, y_train)
'''
# Dự đoán nhãn cho tập kiểm tra
y_pred = knn.predict(X_test)

# Đánh giá độ chính xác
accuracy = accuracy_score(y_test, y_pred)
print("Độ chính xác của mô hình kNN trên tập kiểm tra:", accuracy)
'''

# Đường dẫn tới ảnh cần chuyển đổi
image_path = "test7.png"

# Đọc ảnh và chuyển đổi sang ảnh đen trắng (grayscale)
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Kiểm tra xem ảnh đã được đọc thành công hay chưa
if image is None:
    print("Không thể đọc ảnh!")
    exit()

# Resize ảnh về kích thước 16x16 pixel
resized_image = cv2.resize(image, (16, 16))

# Kiểm tra xem ảnh đã được resize thành công hay chưa
if resized_image is None:
    print("Không thể resize ảnh!")
    exit()

# Chuyển đổi ảnh thành mảng 1D
X_preprocessed = resized_image.flatten()

# Tiền xử lý dữ liệu của ảnh đầu vào để có cùng cấu trúc với dữ liệu UCI Letter
x_box = np.min(np.nonzero(X_preprocessed % 16 != 0))
y_box = np.min(np.nonzero(X_preprocessed // 16 != 0))
width = resized_image.shape[1]
height = resized_image.shape[0]
onpix = np.sum(X_preprocessed)
x_bar = np.mean(np.nonzero(X_preprocessed % 16 != 0))
y_bar = np.mean(np.nonzero(X_preprocessed // 16 != 0))
x2bar = np.mean((np.nonzero(X_preprocessed % 16 != 0)[0]) ** 2)
y2bar = np.mean((np.nonzero(X_preprocessed // 16 != 0)[0]) ** 2)
xybar = np.mean(np.nonzero((X_preprocessed % 16 != 0) & (X_preprocessed // 16 != 0))[0] * np.nonzero((X_preprocessed % 16 != 0) & (X_preprocessed // 16 != 0))[0])
#x2ybr = np.mean((np.nonzero(X_preprocessed % 16 != 0)[0] ** 2) * (np.nonzero(X_preprocessed // 16 != 0)[0]))
#x2ybr = np.mean((np.nonzero(X_preprocessed % 16 != 0)[0] ** 2) * (np.nonzero(X_preprocessed // 16 != 0)[0] ** 2))
#x2ybr = np.mean((np.nonzero(X_preprocessed % 16 != 0)[0] ** 2) * (np.nonzero(X_preprocessed // 16 != 0)[0] ** 3))
#x2ybr = np.mean((np.nonzero(X_preprocessed % 16 != 0)[0] ** 2) * (np.nonzero(X_preprocessed // 16 != 0)[0] ** 2))
#x2ybr = np.mean((np.nonzero(X_preprocessed[:, 8] != 0)[0] ** 2) * (np.nonzero(X_preprocessed[:, 9] != 0)[0]))
#x2ybr = np.mean((np.nonzero(X_preprocessed[:, 8] != 0)[0] ** 2) * (np.nonzero(X_preprocessed[:, 9] != 0)[0] ** 2))
x2ybr = np.mean((np.nonzero(X_preprocessed[:, 7] != 0)[0] ** 2) * (np.nonzero(X_preprocessed[:, 8] != 0)[0] ** 2))

xy2br = np.mean(np.nonzero(X_preprocessed % 16 != 0)[0] * (np.nonzero(X_preprocessed // 16 != 0)[0]) ** 2)
x_ege = np.sum(np.diff(X_preprocessed.reshape(16, 16), axis=1))
xegvy = np.corrcoef(X_preprocessed % 16, X_preprocessed // 16)[0, 1]
y_ege = np.sum(np.diff(X_preprocessed.reshape(16, 16), axis=0))
yegvx = np.corrcoef(X_preprocessed // 16, X_preprocessed % 16)[0, 1]

# Gộp các đặc trưng thành một mảng 1D
X_preprocessed = np.array([x_box, y_box, width, height, onpix, x_bar, y_bar, x2bar, y2bar, xybar, x2ybr, xy2br, x_ege, xegvy, y_ege, yegvx])
# Tiền xử lý dữ liệu của ảnh đầu vào để có cùng cấu trúc với dữ liệu UCI Letter
X_preprocessed = np.expand_dims(X_preprocessed, axis=0)

# Chuẩn hóa dữ liệu
X_preprocessed = (X_preprocessed - np.mean(X)) / np.std(X)

# Tạo mô hình KNN và huấn luyện trên dữ liệu UCI Letter
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X, y)

# Dự đoán nhãn cho ảnh đầu vào
predicted_label = knn_model.predict(X_preprocessed)

# In kết quả
print("Nhãn dự đoán cho ảnh: {}".format(predicted_label[0]))