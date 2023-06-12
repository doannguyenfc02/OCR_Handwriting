import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, BatchNormalization, ReLU, GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
import tensorflow as tf

def ghostnet(input_shape, num_classes):
    inputs = tf.keras.Input(shape=input_shape)
    
    # Convolutional stem
    x = Conv2D(16, kernel_size=3, strides=2, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    # Ghost module
    x = ghost_module(x, 16, ratio=2)
    
    # Depthwise separable convolution blocks
    x = depthwise_block(x, 16, strides=1)
    x = depthwise_block(x, 24, strides=2)
    x = depthwise_block(x, 24, strides=1)
    x = depthwise_block(x, 40, strides=2)
    x = depthwise_block(x, 40, strides=1)
    x = depthwise_block(x, 80, strides=2)
    x = depthwise_block(x, 80, strides=1)
    x = depthwise_block(x, 96, strides=1)
    
    # Classifier head
    x = GlobalAveragePooling2D()(x)
    x = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, x)
    
    return model

def ghost_module(inputs, filters, ratio=2):
    # Pointwise convolution
    pointwise_conv = Conv2D(filters, kernel_size=1)(inputs)
    pointwise_conv = BatchNormalization()(pointwise_conv)
    pointwise_conv = ReLU()(pointwise_conv)
    
    # Ghost branch
    ghost_branch = DepthwiseConv2D(kernel_size=3, strides=1, padding='same')(inputs)
    ghost_branch = BatchNormalization()(ghost_branch)
    ghost_branch = ReLU()(ghost_branch)
    
    ghost_branch = Conv2D(int(filters / ratio), kernel_size=1)(ghost_branch)
    ghost_branch = BatchNormalization()(ghost_branch)
    ghost_branch = ReLU()(ghost_branch)
    
    # Concatenate pointwise convolution and ghost branch
    x = tf.keras.layers.Concatenate()([pointwise_conv, ghost_branch])
    
    return x

def depthwise_block(inputs, filters, strides=1):
    x = DepthwiseConv2D(kernel_size=3, strides=strides, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    x = Conv2D(filters, kernel_size=1)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    return x

# Đọc dữ liệu từ tập train và test
train_data = pd.read_csv('mnist_train.csv')
test_data = pd.read_csv('mnist_test.csv')

# Tách features và labels
X_train = train_data.drop('label', axis=1).values
y_train = train_data['label'].values
X_test = test_data.drop('label', axis=1).values
y_test = test_data['label'].values

# Tiêu chuẩn hóa giá trị pixel về khoảng [0, 1]
X_train = X_train / 255.0
X_test = X_test / 255.0

# Reshape features thành (số mẫu, chiều rộng, chiều cao, số kênh)
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# One-hot encoding cho labels
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Chia tập dữ liệu huấn luyện thành tập huấn luyện và tập đánh giá
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Xây dựng mô hình GhostNet
input_shape = (28, 28, 1)
num_classes = 10
model = ghostnet(input_shape, num_classes)

# Compile mô hình
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Huấn luyện mô hình
model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_val, y_val))

# Đánh giá mô hình trên tập kiểm tra
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("Độ chính xác trên tập kiểm tra: ", test_accuracy)

