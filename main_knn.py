import csv
import cv2
import random
import math
import numpy as np
from PIL import Image


# Đọc dữ liệu từ file CSV
data = []
with open('letter.csv', 'r') as file:
    lines = file.readlines()
    for line in lines[1:]:
        row = line.strip().split(',')
        data.append(row)

# Chia dữ liệu thành features (X) và labels (y)
X = []
y = []
for row in data:
    X.append([int(x) for x in row[1:]])
    y.append(row[0])

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train = []
y_train = []
X_test = []
y_test = []
train_size = 0.8  # Tỷ lệ tập huấn luyện
for i in range(len(X)):
    if random.random() < train_size:
        X_train.append(X[i])
        y_train.append(y[i])
    else:
        X_test.append(X[i])
        y_test.append(y[i])

# Định nghĩa hàm tính khoảng cách Euclidean giữa hai điểm
def euclidean_distance(x1, x2):
    distance = 0
    for i in range(len(x1)):
        distance += (x1[i] - x2[i]) ** 2
    return math.sqrt(distance)




# Định nghĩa hàm dự đoán nhãn cho một điểm dữ liệu dựa trên k láng giềng gần nhất
def predict_knn(X_train, y_train, x_test, k):
    distances = []
    for i in range(len(X_train)):
        distance = euclidean_distance(X_train[i], x_test)
        distances.append((distance, y_train[i]))
    distances.sort(key=lambda x: x[0])  # Sắp xếp tăng dần theo khoảng cách
    neighbors = distances[:k]  # Chọn k láng giềng gần nhất
    counts = {}
    for neighbor in neighbors:
        label = neighbor[1]
        if label in counts:
            counts[label] += 1
        else:
            counts[label] = 1
    predicted_label = max(counts, key=counts.get)  # Chọn nhãn xuất hiện nhiều nhất
    return predicted_label

# Dự đoán nhãn cho tập kiểm tra
y_pred = []
for i in range(len(X_test)):
    predicted_label = predict_knn(X_train, y_train, X_test[i], k=5)
    y_pred.append(predicted_label)

# Đánh giá độ chính xác
correct_predictions = 0
for i in range(len(y_test)):
    if y_test[i] == y_pred[i]:
        correct_predictions += 1
accuracy = correct_predictions / len(y_test)
print("Độ chính xác của mô hình kNN trên tập kiểm tra:", accuracy)



#2
def calculate_x_box(image_path):
    # Mở ảnh với PIL
    image = Image.open(image_path)

    # Chuyển đổi ảnh thành ảnh đen trắng (binary image)
    binary_image = image.convert('1')

    # Tính giá trị x-box
    pixels = binary_image.load()
    x_min = 16
    x_max = 0

    for y in range(16):
        for x in range(16):
            pixel = pixels[x, y]
            if pixel == 255:  # Nếu pixel là pixel bật
                if x < x_min:
                    x_min = x
                if x > x_max:
                    x_max = x

    # Tính giá trị x-box
    x_box = x_max - x_min

    return x_box
#3
def calculate_y_box(image_path):
    # Mở ảnh với PIL
    image = Image.open(image_path)

    # Chuyển đổi ảnh thành ảnh đen trắng (binary image)
    binary_image = image.convert('1')

    # Tính giá trị y-box
    pixels = binary_image.load()
    y_min = 16
    y_max = 0

    for y in range(16):
        for x in range(16):
            pixel = pixels[x, y]
            if pixel == 255:  # Nếu pixel là pixel bật
                if y < y_min:
                    y_min = y
                if y > y_max:
                    y_max = y

    # Tính giá trị y-box
    y_box = y_max - y_min

    return y_box
#4
def calculate_width(image_path):
    # Mở ảnh với PIL
    image = Image.open(image_path)

    # Chuyển đổi ảnh thành ảnh đen trắng (binary image)
    binary_image = image.convert('1')

    # Tính giá trị width
    pixels = binary_image.load()
    x_min = 16
    x_max = 0

    for y in range(16):
        for x in range(16):
            pixel = pixels[x, y]
            if pixel == 255:  # Nếu pixel là pixel bật
                if x < x_min:
                    x_min = x
                if x > x_max:
                    x_max = x

    # Tính giá trị width
    width = x_max - x_min + 1

    return width
#5
def calculate_high(image_path):
    # Mở ảnh với PIL
    image = Image.open(image_path)

    # Chuyển đổi ảnh thành ảnh đen trắng (binary image)
    binary_image = image.convert('1')

    # Tính giá trị high
    pixels = binary_image.load()
    y_min = 16
    y_max = 0

    for y in range(16):
        for x in range(16):
            pixel = pixels[x, y]
            if pixel == 255:  # Nếu pixel là pixel bật
                if y < y_min:
                    y_min = y
                if y > y_max:
                    y_max = y

    # Tính giá trị high
    high = y_max - y_min + 1

    return high
#6
def calculate_onpix(image_path):
    # Mở ảnh với PIL
    image = Image.open(image_path)

    # Chuyển đổi ảnh thành ảnh đen trắng (binary image)
    binary_image = image.convert('1')

    # Tính giá trị onpix
    pixels = binary_image.load()
    onpix = 0

    for y in range(16):
        for x in range(16):
            pixel = pixels[x, y]
            if pixel == 255:  # Nếu pixel là pixel bật
                onpix += 1

    return onpix
#7
def calculate_x_bar(image_path):
    # Mở ảnh với PIL
    image = Image.open(image_path)

    # Chuyển đổi ảnh thành ảnh đen trắng (binary image)
    binary_image = image.convert('1')

    # Tính giá trị x_bar
    pixels = binary_image.load()
    onpix = 0
    x_sum = 0

    for y in range(16):
        for x in range(16):
            pixel = pixels[x, y]
            if pixel == 255:  # Nếu pixel là pixel bật
                onpix += 1
                x_sum += x

    if onpix == 0:
        return 0

    x_bar = x_sum / onpix

    return x_bar
#8
def calculate_y_bar(image_path):
    # Mở ảnh với PIL
    image = Image.open(image_path)

    # Chuyển đổi ảnh thành ảnh đen trắng (binary image)
    binary_image = image.convert('1')

    # Tính giá trị y_bar
    pixels = binary_image.load()
    onpix = 0
    y_sum = 0

    for y in range(16):
        for x in range(16):
            pixel = pixels[x, y]
            if pixel == 255:  # Nếu pixel là pixel bật
                onpix += 1
                y_sum += y

    if onpix == 0:
        return 0

    y_bar = y_sum / onpix

    return y_bar
#9
def calculate_x2bar(image_path):
    # Mở ảnh với PIL
    image = Image.open(image_path)

    # Chuyển đổi ảnh thành ảnh đen trắng (binary image)
    binary_image = image.convert('1')

    # Tính giá trị x2bar
    pixels = binary_image.load()
    onpix = 0
    x_sum = 0
    x2_sum = 0

    for y in range(16):
        for x in range(16):
            pixel = pixels[x, y]
            if pixel == 255:  # Nếu pixel là pixel bật
                onpix += 1
                x_sum += x
                x2_sum += x**2

    if onpix == 0:
        return 0

    x_bar = x_sum / onpix
    x2bar = (x2_sum / onpix) - x_bar**2

    return x2bar
#10
def calculate_y2bar(image_path):
    # Mở ảnh với PIL
    image = Image.open(image_path)

    # Chuyển đổi ảnh thành ảnh đen trắng (binary image)
    binary_image = image.convert('1')

    # Tính giá trị y2bar
    pixels = binary_image.load()
    onpix = 0
    y_sum = 0
    y2_sum = 0

    for y in range(16):
        for x in range(16):
            pixel = pixels[x, y]
            if pixel == 255:  # Nếu pixel là pixel bật
                onpix += 1
                y_sum += y
                y2_sum += y**2

    if onpix == 0:
        return 0

    y_bar = y_sum / onpix
    y2bar = (y2_sum / onpix) - y_bar**2

    return y2bar

#11
def calculate_xybar(image_path):
    # Mở ảnh với PIL
    image = Image.open(image_path)

    # Chuyển đổi ảnh thành ảnh đen trắng (binary image)
    binary_image = image.convert('1')

    # Tính giá trị xybar
    pixels = binary_image.load()
    onpix = 0
    x_sum = 0
    y_sum = 0
    xy_sum = 0

    for y in range(16):
        for x in range(16):
            pixel = pixels[x, y]
            if pixel == 255:  # Nếu pixel là pixel bật
                onpix += 1
                x_sum += x
                y_sum += y
                xy_sum += x * y

    if onpix == 0:
        return 0

    x_bar = x_sum / onpix
    y_bar = y_sum / onpix
    xybar = (xy_sum / onpix) - (x_bar * y_bar)

    return xybar
#12
def calculate_x2ybr(image_path):
    # Mở ảnh với PIL
    image = Image.open(image_path)

    # Chuyển đổi ảnh thành ảnh đen trắng (binary image)
    binary_image = image.convert('1')

    # Tính giá trị x2ybr
    pixels = binary_image.load()
    onpix = 0
    x_sum = 0
    y_sum = 0
    x2y_sum = 0

    for y in range(16):
        for x in range(16):
            pixel = pixels[x, y]
            if pixel == 255:  # Nếu pixel là pixel bật
                onpix += 1
                x_sum += x
                y_sum += y
                x2y_sum += x * x * y

    if onpix == 0:
        return 0

    x_bar = x_sum / onpix
    y_bar = y_sum / onpix
    x2ybr = (x2y_sum / onpix) - (x_bar * y_bar * y_bar)

    return x2ybr

#13
def calculate_xy2br(image_path):
    # Mở ảnh với PIL
    image = Image.open(image_path)

    # Chuyển đổi ảnh thành ảnh đen trắng (binary image)
    binary_image = image.convert('1')

    # Tính giá trị xy2br
    pixels = binary_image.load()
    onpix = 0
    x_sum = 0
    y_sum = 0
    xy2_sum = 0

    for y in range(16):
        for x in range(16):
            pixel = pixels[x, y]
            if pixel == 255:  # Nếu pixel là pixel bật
                onpix += 1
                x_sum += x
                y_sum += y
                xy2_sum += x * y * y

    if onpix == 0:
        return 0

    x_bar = x_sum / onpix
    y_bar = y_sum / onpix
    xy2br = (xy2_sum / onpix) - (x_bar * x_bar * y_bar)

    return xy2br

#14
def calculate_x_ege(image_path):
    # Mở ảnh với PIL
    image = Image.open(image_path)

    # Chuyển đổi ảnh thành ảnh đen trắng (binary image)
    binary_image = image.convert('1')

    # Tính giá trị x-ege
    pixels = binary_image.load()
    x_ege = 0

    for y in range(16):
        for x in range(16 - 1):
            pixel = pixels[x, y]
            next_pixel = pixels[x + 1, y]
            if pixel != next_pixel:  # Nếu có sự khác biệt giữa hai pixel kế tiếp
                x_ege += 1

    return x_ege
#15
def calculate_xegvy(image_path):
    # Mở ảnh với PIL
    image = Image.open(image_path)

    # Chuyển đổi ảnh thành ảnh đen trắng (binary image)
    binary_image = image.convert('1')

    # Tính giá trị xegvy
    pixels = binary_image.load()
    x_ege = 0
    y_sum = 0

    for y in range(16):
        for x in range(16 - 1):
            pixel = pixels[x, y]
            next_pixel = pixels[x + 1, y]
            if pixel != next_pixel:  # Nếu có sự khác biệt giữa hai pixel kế tiếp
                x_ege += 1
                y_sum += y

    if x_ege == 0:
        return 0

    xegvy = y_sum / x_ege

    return xegvy
#16
def calculate_y_ege(image_path):
    # Mở ảnh với PIL
    image = Image.open(image_path)

    # Chuyển đổi ảnh thành ảnh đen trắng (binary image)
    binary_image = image.convert('1')

    # Tính giá trị y-ege
    pixels = binary_image.load()
    y_ege = 0

    for x in range(16):
        for y in range(16 - 1):
            pixel = pixels[x, y]
            next_pixel = pixels[x, y + 1]
            if pixel != next_pixel:  # Nếu có sự khác biệt giữa hai pixel kế tiếp
                y_ege += 1

    return y_ege
#17
def calculate_yegvx(image_path):
    # Mở ảnh với PIL
    image = Image.open(image_path)

    # Chuyển đổi ảnh thành ảnh đen trắng (binary image)
    binary_image = image.convert('1')

    # Tính giá trị yegvx
    pixels = binary_image.load()
    y_ege = 0
    x_sum = 0

    for x in range(16):
        for y in range(16 - 1):
            pixel = pixels[x, y]
            next_pixel = pixels[x, y + 1]
            if pixel != next_pixel:  # Nếu có sự khác biệt giữa hai pixel kế tiếp
                y_ege += 1
                x_sum += x

    if y_ege == 0:
        return 0

    yegvx = x_sum / y_ege

    return yegvx

image_path = 'test.png'
# Tiền xử lý ảnh
x_box= calculate_x_box(image_path)      #2
y_box= calculate_y_box(image_path)      #3
width= calculate_width(image_path)    #4
high=  calculate_high(image_path)     #5
onpix =    calculate_onpix(image_path)#6
x_bar =   calculate_x_bar(image_path)  #7
y_bar = calculate_y_box(image_path)    #8
x2bar = calculate_x2bar(image_path)    #9
y2bar = calculate_y2bar(image_path)    #10
xybar = calculate_xybar(image_path)    #11
x2ybr = calculate_x2ybr(image_path)    #12
xy2br=   calculate_xy2br(image_path)   #13
x_ege =  calculate_x_ege(image_path)   #14
xegvy=  calculate_xegvy(image_path)    #15
y_ege =  calculate_y_ege(image_path)   #16
yegvx=   calculate_yegvx(image_path)   #17




# Gộp các đặc trưng thành một mảng 1D
input_data = np.array([x_box, y_box, width, high, onpix, x_bar, y_bar, x2bar, y2bar, xybar, x2ybr, xy2br, x_ege, xegvy, y_ege, yegvx])

# Dự đoán nhãn
k = 5  # Số láng giềng gần nhất
predicted_label = predict_knn(X_train, y_train, input_data, k)
print("Nhãn dự đoán cho ảnh {}: {}".format(image_path, predicted_label))