import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, DepthwiseConv2D, GlobalAveragePooling2D, Dense
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

def ghostnet_block(inputs, expansion, channels, stride):
    x = Conv2D(expansion * inputs.shape[-1], kernel_size=(1, 1))(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = DepthwiseConv2D(kernel_size=(3, 3), strides=(stride, stride), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Conv2D(channels, kernel_size=(1, 1))(x)
    x = BatchNormalization()(x)
    
    if stride == 1 and inputs.shape[-1] == channels:
        x = tf.keras.layers.add([x, inputs])
    
    return x

def GhostNet(input_shape, num_classes):
    inputs = tf.keras.Input(shape=input_shape)
    
    x = Conv2D(16, kernel_size=(3, 3), strides=(2, 2), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = ghostnet_block(x, expansion=16, channels=16, stride=1)
    x = ghostnet_block(x, expansion=48, channels=24, stride=2)
    x = ghostnet_block(x, expansion=72, channels=24, stride=1)
    x = ghostnet_block(x, expansion=72, channels=40, stride=2)
    x = ghostnet_block(x, expansion=120, channels=40, stride=1)
    x = ghostnet_block(x, expansion=240, channels=80, stride=2)
    x = ghostnet_block(x, expansion=240, channels=80, stride=1)
    x = ghostnet_block(x, expansion=480, channels=112, stride=1)
    x = ghostnet_block(x, expansion=672, channels=160, stride=2)
    
    x = Conv2D(960, kernel_size=(1, 1))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = GlobalAveragePooling2D()(x)
    x = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=x)
    return model

# Định dạng dữ liệu đầu vào
input_shape = (28, 28, 1)
num_classes = 10

# Load dữ liệu MNIST
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Chia tập train thành tập train và tập validation
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Reshape và chuẩn hóa dữ liệu
X_train = X_train.reshape(-1, 28, 28, 1) / 255.0
X_val = X_val.reshape(-1, 28, 28, 1) / 255.0
X_test = X_test.reshape(-1, 28, 28, 1) / 255.0

# One-hot encoding cho labels
y_train = to_categorical(y_train, num_classes)
y_val = to_categorical(y_val, num_classes)
y_test = to_categorical(y_test, num_classes)

# Xây dựng mô hình GhostNet
model = GhostNet(input_shape, num_classes)

# Compile mô hình
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Huấn luyện mô hình
model.fit(X_train, y_train, batch_size=128, epochs=10, validation_data=(X_val, y_val))

# Đánh giá mô hình trên tập test
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("Độ chính xác trên tập test: ", test_accuracy)

