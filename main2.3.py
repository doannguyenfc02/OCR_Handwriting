import csv
import numpy as np
import cv2
from matplotlib import pyplot as plt
from numpy.core.fromnumeric import sort

"""ham tinh so pixel khac nhau giua 2 hinh"""
def different(test_sample, train_sample):
    total_different = 0  #khởi tạo biến để đếm số lượng pixel giữa hai hình ảnh
    for index in range(28*28):
         '''
         so sánh giá trị của pixel tại chỉ số index của hai hình ảnh
         - nếu giá trị pixel tại chỉ số index của test_sample nằm trong khoảng giá trị của train_sample (+-10) thì không tính khác nhau và bỏ qua
         - ngược lại thì tawng biến đếm total_diffenent lên 1
         '''
         if ((test_sample[0][index]<=(train_sample[0][index]+20)) and (test_sample[0][index]>=(train_sample[0][index]-20))):
            ''''''
         else:
            total_different+=1
    return total_different 

"""sort theo so pixel khac nhau"""
#sắp xếp láng liềng theo thứ tự tăng dần khoảng các euclid
def sortNeighbors(neighbor):
   return neighbor['dis']


"""KNN. K la so diem lan can muon lay"""
def KNN(dataset,test,K):
   k_list=[]  #khởi tạo danh sách rỗng để lưu các láng giềng gần nhất
   '''
   -Lặp qua mỗi mẫu của dataset
   -Với mỗi mẫu ta tính khoảng cách với mẫu test bằng cách sử dụng hàm diffent
   -sau đó lưu kết quả vào k_list
   '''
   for row in dataset: 
         #thêm một từ điển vào ds k_list, trong đó label tương ứng với nhãm, dis tương ứng với khaorng cách giữa mẫu trong dataset và mẫu test
         k_list.append({'label':row[0],'dis':different(test,[row[1:]])})
   k_list.sort(key=sortNeighbors) #sắp xếp danh sách k_list theo thứ tự tăng dần
   k_list=k_list[:K]   #lấy ra k phần tử đầu tiên của danh sách k_list
   m_list=[]           #khởi tạo một mảng để lưu các nhãn của k phần tử 
   #thêm lần lượt các nhãn tương ứng của k phần tử vào m_list
   for i in range(K):
      m_list.append(k_list[i]['label'])

   return max(set(m_list),key=m_list.count)   #trả về nhãn xuất hiện nhiều nhất trong k_list và chuyển thành ASCII





#đọc dữ liệu từ tập new.csv và dataset.csv
data = csv.DictReader(open("mnist_train.csv"))    #đọc file mnist_train.csv
datatest = csv.DictReader(open("mnist_test4.csv"))  


data_images=[]  #khởi tạo danh sách để  lưu dữ liệu hình ảnh
#duyệt qua tất cả các dữ liệu trong data
for row in data:
    image = []   #khởi tạo một list để lưu dữ liệu 1 ảnh
    image.append(int(row["label"]))    #lưu nhãn của ảnh vào image
    #lưu tất cả các giá trị pixel của ảnh vào image
    for index in range(0, 28**2):
        image.append(int(row['pixel' + str(index)]))
    data_images.append(image)   #thêm image vào danh sách data_images. sau khi kết thúc vòng lặp data_images sẽ chứa toàn bộ dữ liệu huấn luyện

"""chay voi mau viet tay bat ki"""
imagetest=cv2.imread('so.png',0)   #đọc ảnh cần kiêm tra, chuyển xám
#imagetest=imagetest.reshape(1,28*28)   #chuyển các pixel về 1 chiều
predicted_label=KNN(data_images,imagetest,5)
print("Predicted label: ", predicted_label)
