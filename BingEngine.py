import os

import torch
from PIL import Image
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim

# 图片预处理模式
data_transforms = {
    # train训练集处理方式：随机裁剪、水平翻转、转换为张量、归一化
    'train': transforms.Compose([
        transforms.Resize((224, 244)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    # val验证集处理方式：转换为张量、归一化
    'val': transforms.Compose([
        transforms.Resize((224, 244)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

def main():
    # 检查GPU是否可用 若可用则使用GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 加载数据
    # 数据集总路径
    data_dir = 'data/dataset'
    # 读取图片数据集 它是一个字典类型 包含训练集和验证集 键为train和val 值为数据集（通过ImageFolder方法第一个参数是文件夹路径，第二个参数是预处理模式）
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
    # 通过图片数据集获得对应的数据加载器
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=4) for x in ['train', 'val']}
    # 通过图片数据集数据集大小
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    # 使用预训练的模型resnet50
    model = models.resnet50(pretrained=True)

    # 冻结模型参数 不进行梯度更新 仅训练最后一层 以适应我们的分类任务
    for param in model.parameters():
        param.requires_grad = False

    # 替换最后一层以适应我们的分类任务 将最后一层的输出特征数量转换为2 （原来的最后一层被定义这个最新的全连接层代替了）
    num_ftrs = model.fc.in_features
    # 将最后一层的输出特征数量转换为2 （新定义的全连接层）
    model.fc = nn.Linear(num_ftrs, 2)

    # 将模型移动到GPU
    model = model.to(device)

    # 定义损失函数
    criterion = nn.CrossEntropyLoss()
    # 定义优化器
    optimizer = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)

    # 训练模型 5轮
    for epoch in range(5):
        # 每轮都是先训练再验证
        for phase in ['train', 'val']:
            # 校准模式
            if phase == 'train':
                # 训练模式
                model.train()
            else:
                # 验证模式
                model.eval()

            # 记录损失和准确率
            running_loss = 0.0
            running_corrects = 0

            # 遍历数据加载器
            for inputs, labels in dataloaders[phase]:
                # 将输入和标签移动到GPU
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 梯度清零
                optimizer.zero_grad()

                # 前向传播
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
    torch.save(model.state_dict(), 'model1.pth')


def test():
    params = torch.load('model1.pth')

    model = models.resnet50()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model.load_state_dict(params, strict=False)
    model.eval()

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((224, 244)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 加载图像
    image = Image.open('data/dataset/end/Dog/55.jpg')
    image = transform(image).unsqueeze(0)

    # 预测
    with torch.no_grad():
        output = model(image)
        print(output)
        _, predicted = torch.max(output, 1)

    print('Predicted:', '猫' if predicted.item() == 0 else '狗')

if __name__ == '__main__':
    main()
