import os

import torch
from PIL import Image
from torch import optim

import torch.nn as nn
import torch.nn.functional as F

from torchvision.transforms import transforms


# 1.准备工作
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 53 * 53, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 53 * 53)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = Net().cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

# 2.训练集
transform = transforms.Compose([
    transforms.Resize([int(128 * 224 / 112), int(128 * 224 / 112)]),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

train_features = []
train_labels = []
path = "data/dataset/train/Dog"  # 替换为你的文件夹路径
files = os.listdir(path)  # 得到文件夹下的所有文件名称
for file in files:  # 遍历文件夹
    if not os.path.isdir(file):  # 判断是否是文件夹，不是文件夹才打开
        way = path + "/" + file
        img = Image.open(way)
        img = img.resize((224, 224)).convert('RGB')
        img = transform(img).unsqueeze(0)
        train_features.append(img)
        train_labels.append(torch.tensor([0]))

path = "data/dataset/train/Cat"  # 替换为你的文件夹路径
files = os.listdir(path)  # 得到文件夹下的所有文件名称
for file in files:  # 遍历文件夹
    if not os.path.isdir(file):  # 判断是否是文件夹，不是文件夹才打开
        way = path + "/" + file
        img = Image.open(way)
        img = img.resize((224, 224)).convert('RGB')
        img = transform(img).unsqueeze(0)
        train_features.append(img)
        train_labels.append(torch.tensor([1]))

train_features = [img.cuda() for img in train_features]  # Move all image tensors to GPU
train_labels = [label.cuda() for label in train_labels]  # Move all label tensors to GPU

# 3.测试集
test_features = []
test_labels = []
path = "data/dataset/test/Dog"  # 替换为你的文件夹路径
files = os.listdir(path)  # 得到文件夹下的所有文件名称
for file in files:  # 遍历文件夹
    if not os.path.isdir(file):  # 判断是否是文件夹，不是文件夹才打开
        way = path + "/" + file
        img = Image.open(way)
        img = img.resize((224, 224)).convert('RGB')
        img = transform(img).unsqueeze(0)
        test_features.append(img)
        test_labels.append(torch.tensor([0]))

path = "data/dataset/test/Cat"  # 替换为你的文件夹路径
files = os.listdir(path)  # 得到文件夹下的所有文件名称
for file in files:  # 遍历文件夹
    if not os.path.isdir(file):  # 判断是否是文件夹，不是文件夹才打开
        way = path + "/" + file
        img = Image.open(way)
        img = img.resize((224, 224)).convert('RGB')
        img = transform(img).unsqueeze(0)
        test_features.append(img)
        test_labels.append(torch.tensor([1]))

test_features = [img.cuda() for img in test_features]  # Move all image tensors to GPU
test_labels = [label.cuda() for label in test_labels]  # Move all label tensors to GPU


# 4.训练模型
for i in range(15):
    optimizer.zero_grad()
    # 训练集表现
    success = 0
    for j, (feature, label) in enumerate(zip(train_features, train_labels)):
        predict = model(feature)
        # 计算损失
        loss = criterion(predict, label)
        # 反向传播
        loss.backward()
        # 梯度下降
        optimizer.step()
        #print('第{}次训练，第{}个样本，损失为：{}，准确：{}'.format(i, j, loss.item(), predict.argmax().item() == label.item()))
        if predict.argmax().item() == label.item():
            success += 1
    print('第{}次训练，训练集准确率为：{}'.format(i, success / len(train_features)))

    # 测试集表现
    success = 0
    for j, (feature, label) in enumerate(zip(test_features, test_labels)):
        predict = model(feature)
        #print(predict, label)
        #print('第{}次训练，第{}个样本，准确：{}'.format(i, j, predict.argmax().item() == label.item()))
        if predict.argmax().item() == label.item():
            success += 1
    print('第{}次训练，测试集准确率为：{}'.format(i, success / len(test_features)))

# 5.保存模型
torch.save(model.state_dict(), 'model.pth')




