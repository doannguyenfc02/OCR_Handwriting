import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Đọc dữ liệu từ tập train
train_data = pd.read_csv('mnist_train.csv')

# Tách features và labels
X_train = train_data.drop('label', axis=1).values
y_train = train_data['label'].values

# Tiêu chuẩn hóa giá trị pixel về khoảng [0, 1]
X_train = X_train / 255.0

# Reshape features thành (số mẫu, chiều rộng, chiều cao, số kênh)
X_train = X_train.reshape(-1, 28, 28, 1)

# One-hot encoding cho labels
y_train = to_categorical(y_train)

# Chia tập dữ liệu huấn luyện thành tập huấn luyện và tập đánh giá
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Xây dựng mô hình CNN
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Compile mô hình
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Huấn luyện mô hình trên tập huấn luyện
model.fit(X_train, y_train, batch_size=128, epochs=10, validation_data=(X_val, y_val))

# Lưu mô hình
model.save('model_cnn_mnist.h5')

# Đánh giá mô hình trên tập kiểm tra
test_data = pd.read_csv('mnist_test.csv')
X_test = test_data.drop('label', axis=1).values
y_test = test_data['label'].values
X_test = X_test / 255.0
X_test = X_test.reshape(-1, 28, 28, 1)
y_test = to_categorical(y_test)

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("Độ chính xác trên tập kiểm tra: ", test_accuracy)
