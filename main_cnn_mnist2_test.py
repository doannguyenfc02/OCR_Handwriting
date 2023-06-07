import numpy as np
from keras.models import load_model
from PIL import Image

# Load mô hình đã lưu
model = load_model('model_cnn_mnist.h5')

# Đường dẫn đến ảnh cần đánh giá
image_path = 'test.png'

# Đọc ảnh và tiền xử lý
image = Image.open(image_path).convert('L')
image = image.resize((28, 28))
image = np.array(image)
image = image.reshape(1, 28, 28, 1)
image = image / 255.0

# Dự đoán nhãn của ảnh
prediction = model.predict(image)
label = np.argmax(prediction)

# Hiển thị kết quả
print("Predicted label: ", label)
