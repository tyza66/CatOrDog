import os
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image

class CustomImageFolder(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # 创建一个字典，将类别的名称映射到你想要的编号
        self.class_to_idx = {'cat': 0, 'dog': 1}

        # 读取所有的图片和对应的类别
        self.imgs = []
        for class_name in os.listdir(root_dir):
            class_dir = os.path.join(root_dir, class_name)
            for img_name in os.listdir(class_dir):
                self.imgs.append((os.path.join(class_dir, img_name), self.class_to_idx[class_name]))

    def __getitem__(self, idx):
        img_path, label = self.imgs[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)

# 使用自定义的数据集
data_transform = transforms.Compose([
    transforms.Resize((224, 244)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
dataset = CustomImageFolder(root_dir='path_to_your_data', transform=data_transform)
