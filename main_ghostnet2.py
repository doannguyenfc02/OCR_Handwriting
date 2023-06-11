import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
from torchvision.models import ghostnet

# Định nghĩa mô hình GhostNet
class GhostNet(nn.Module):
    def __init__(self, num_classes):
        super(GhostNet, self).__init__()
        self.model = ghostnet(pretrained=False, num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

# Định nghĩa lớp dữ liệu tùy chỉnh
class CustomDataset(Dataset):
    def __init__(self, file_path):
        self.data = self.load_data(file_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = self.data[index][1:]  # Bỏ qua cột đầu tiên (nhãn)
        image = torch.tensor(image, dtype=torch.float32) / 255.0
        image = image.view(1, 28, 28)  # Định dạng lại thành kích thước 28x28

        label = self.data[index][0]  # Nhãn

        return image, label

    def load_data(self, file_path):
        # Đọc dữ liệu từ tệp CSV và chuyển thành list
        with open(file_path, 'r') as file:
            lines = file.readlines()
        data = [line.strip().split(',') for line in lines]
        return data

# Chuẩn bị dữ liệu
train_dataset = CustomDataset('mnist_train.csv')
test_dataset = CustomDataset('mnist_test.csv')

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Xây dựng mô hình
model = GhostNet(num_classes=10)  # 10 là số lượng nhãn trong MNIST
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Huấn luyện mô hình
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")

# Đánh giá mô hình
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Accuracy on test set: {accuracy}%")



