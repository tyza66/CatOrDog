import torch
from basetrainer.utils import setup_config
from torchvision.transforms import transforms
from PIL import Image

from Engine import ClassificationTrainer, get_parser
from torchvision.datasets import MNIST, ImageFolder

def test():
    parser = get_parser()
    cfg = setup_config.parser_config(parser.parse_args(), cfg_updata=True)
    params = torch.load("./model1/best_model_099_100.0000.pth")
    print(params.keys())
    ct = ClassificationTrainer(cfg)
    model = ct.build_model(cfg)
    model.load_state_dict(params, strict=False)
    transform = transforms.Compose([
        transforms.Resize([int(128 * cfg.input_size[1] / 112), int(128 * cfg.input_size[0] / 112)]),
        transforms.CenterCrop([cfg.input_size[1], cfg.input_size[0]]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    dataset = ImageFolder(root="data/dataset/test", transform=transform)
    loader = ct.build_dataloader(dataset, cfg.batch_size, cfg.num_workers, phase="test",
                                   shuffle=False, pin_memory=False, drop_last=False, distributed=False)
    model.eval()
    for data, target in loader:
        result = model(data)
        print(result)


def test1():
    # 加载模型
    parser = get_parser()

    cfg = setup_config.parser_config(parser.parse_args(), cfg_updata=True)
    ct = ClassificationTrainer(cfg)
    model = ct.build_model(cfg)
    state_dict = torch.load("./model1/best_model_027_100.0000.pth")  # 你的模型参数文件
    model.load_state_dict(state_dict, strict=False)

    # 预处理数据
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize([int(128 * cfg.input_size[1] / 112), int(128 * cfg.input_size[0] / 112)]),
        transforms.CenterCrop([cfg.input_size[1], cfg.input_size[0]]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    image = Image.open("data/dataset/end/Dog/dog.1514.jpg").convert('RGB')  # 加载你的图像
    image = transform(image).unsqueeze(0)  # 预处理图像并增加批处理维度

    # 运行模型
    model.eval()  # 设置模型为评估模式
    with torch.no_grad():  # 不需要计算梯度
        output = model(image)

    print(output)
    # 解析输出
    _, predicted = torch.max(output, 1)
    print("Predicted class:", predicted.item())
    # 狗还是猫
    labels = ['Cat', 'Dog']
    print("Predicted class name:", labels[predicted.item()])


if __name__ == '__main__':
    test()
