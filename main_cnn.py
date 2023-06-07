import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical
from PIL import Image
from keras.models import load_model


# 1. Chuẩn bị dữ liệu
data = pd.read_csv('mnist_train.csv')
X = data.iloc[:, 1:].values / 255.0
y = data.iloc[:, 0].values
X = X.reshape(-1, 28, 28, 1)
y = to_categorical(y, num_classes=10)

# 2. Tiền xử lý dữ liệu
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Xây dựng mô hình CNN
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

# 4. Huấn luyện mô hình
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=128, epochs=10, validation_data=(X_test, y_test))
# 5. Đánh giá mô hình
accuracy = model.evaluate(X_test, y_test)[1]
print("Accuracy:", accuracy)

'''
# 6. Nhận dạng ảnh test
image_path = 'test7.png'
image = Image.open(image_path).convert('L')  # Chuyển đổi ảnh thành định dạng grayscale
image = image.resize((28, 28))  # Đặt kích thước ảnh thành 28x28
image_array = np.array(image) / 255.0  # Chuẩn hóa giá trị pixel về khoảng [0, 1]
image_array = image_array.reshape(1, 28, 28, 1)  # Reshape ảnh thành (1, 28, 28, 1)

# Nhận dạng ảnh
prediction = model.predict(image_array)
predicted_label = np.argmax(prediction)

print("Predicted label:", predicted_label)
'''