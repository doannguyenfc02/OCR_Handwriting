import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

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


# Tiêu chuẩn hóa đặc trưng
X_train = X_train / 255.0
X_test = X_test / 255.0

# Reshape lại kích thước hình ảnh
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)




# Xây dựng mô hình CNN
model = tf.keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Biên dịch mô hình
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Huấn luyện mô hình
model.fit(X_train, y_train, epochs=10, batch_size=32)
# Đánh giá mô hình trên tập kiểm tra
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("Độ chính xác trên tập kiểm tra: ", test_accuracy)



from PIL import Image

# Đọc ảnh test.png
test_image = Image.open('test7.png').convert('L')

# Thay đổi kích thước ảnh thành 28x28 pixel
test_image = test_image.resize((28, 28))

# Chuyển ảnh thành mảng numpy
test_image = np.array(test_image)

# Tiêu chuẩn hóa ảnh
test_image = test_image / 255.0

# Reshape ảnh thành (1, 28, 28, 1) để phù hợp với đầu vào mô hình CNN
test_image = test_image.reshape(1, 28, 28, 1)
# Sử dụng mô hình CNN đã huấn luyện để nhận dạng chữ số
predictions = model.predict(test_image)
predicted_label = np.argmax(predictions)

print("Chữ số được nhận dạng trên ảnh: ", predicted_label)

# Lưu mô hình
model.save("model_cnn_mnist.h5")


