"""读取和预处理数据"""

import torch

# DataLoader：将多个样本组成一个批次。
# random_split：按照指定数量划分数据集。
from torch.utils.data import DataLoader, random_split

# FashionMNIST：torchvision 提供的服装图片数据集。
from torchvision.datasets import FashionMNIST

# ToTensor：把图片转换成 PyTorch Tensor。
from torchvision.transforms import ToTensor


def create_dataloaders(batch_size: int):
    """创建训练集、验证集和测试集的数据加载器。

    参数：
        batch_size：每个批次包含的图片数量。

    返回：
        train_loader：训练数据加载器。
        validation_loader：验证数据加载器。
        test_loader：测试数据加载器。
    """

    # 加载 FashionMNIST 的原始训练集，共 60,000 张图片。
    full_train_dataset = FashionMNIST(
        # 将数据下载并保存到当前项目的 data 文件夹。
        root="data",

        # train=True 表示加载官方训练集。
        train=True,

        # 本地没有数据时自动下载。
        download=True,

        # 取出图片时，将其转换成 Tensor。
        transform=ToTensor(),
    )

    # 加载 FashionMNIST 的官方测试集，共 10,000 张图片。
    test_dataset = FashionMNIST(
        root="data",

        # train=False 表示加载官方测试集。
        train=False,

        download=True,
        transform=ToTensor(),
    )

    # 将60,000张原始训练图片划分为训练集和验证集。
    train_dataset, validation_dataset = random_split(
        full_train_dataset,

        # 前55,000张作为训练集，剩下5,000张作为验证集。
        lengths=[55_000, 5_000],

        # 固定随机种子，保证每次划分出的数据相同。
        generator=torch.Generator().manual_seed(42),
    )

    # 创建训练集的数据加载器。
    train_loader = DataLoader(
        train_dataset,

        # 每次取出 batch_size 张图片。
        batch_size=batch_size,

        # 每轮训练前打乱训练数据的顺序。
        shuffle=True,

        # 暂时使用主进程读取数据，Windows下更容易调试。
        num_workers=0,
    )

    # 创建验证集的数据加载器。
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,

        # 验证时不需要打乱数据顺序。
        shuffle=False,

        num_workers=0,
    )

    # 创建测试集的数据加载器。
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,

        # 测试时也不需要打乱数据顺序。
        shuffle=False,

        num_workers=0,
    )

    # 一次返回三个数据加载器。
    return train_loader, validation_loader, test_loader


# 只有直接运行 dataset.py 时，才执行下面的测试代码。
# 如果其他文件通过 import 导入 dataset.py，则不会执行。
if __name__ == "__main__":
    # 调用函数，创建三个数据加载器。
    train_loader, validation_loader, test_loader = create_dataloaders(
        batch_size=64
    )

    # iter() 把数据加载器转换成迭代器。
    # next() 从迭代器中取出第一个批次。
    #
    # 每个批次包含两个Tensor：
    # images：一个批次的图片。
    # labels：每张图片对应的类别编号。
    images, labels = next(iter(train_loader))

    # len(loader) 表示一个完整数据集中有多少个批次。
    print(f"训练批次数量：{len(train_loader)}")
    print(f"验证批次数量：{len(validation_loader)}")
    print(f"测试批次数量：{len(test_loader)}")

    # images.shape 应类似 [64, 1, 28, 28]。
    print(f"图片批次形状：{images.shape}")

    # labels.shape 应类似 [64]，表示64张图片各有一个标签。
    print(f"标签批次形状：{labels.shape}")

    # ToTensor() 默认将图片转换为 float32 类型。
    print(f"图片数据类型：{images.dtype}")

    # DataLoader 默认把数据加载到 CPU。
    print(f"图片所在设备：{images.device}")