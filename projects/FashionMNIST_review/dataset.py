"""
下载并读取 FashionMNIST. 
划分训练集和验证集，为训练、验证和测试分别创建按批次读取数据的 DataLoader。

Dataset：管理单个样本，能够根据索引返回“图像、标签”。
DataLoader：读取 Dataset，将多个样本组合成一个批次。

"""

# Path 提供跨平台的路径拼接和解析功能。
from pathlib import Path

# Generator 用于控制数据划分时使用的随机数。
import torch

# DataLoader 将样本组成批次；random_split 随机划分数据集。
from torch.utils.data import DataLoader, random_split

# FashionMNIST 封装数据下载与读取；ToTensor 将图像转换为张量。
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor

from config import TrainConfig


# tuple[...] 表示返回一个元组，其中依次放置三个 DataLoader。
def build_dataloaders(
    config: TrainConfig,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """返回训练、验证和测试数据加载器。"""

    # __file__ 是当前文件的路径；resolve() 得到绝对路径。
    # parent 取得文件所在目录；Path 的 / 运算符用于拼接路径。
    project_dir = Path(__file__).resolve().parent
    data_dir = project_dir / config.data_dir

    # 将灰度图转为 [1, 28, 28] 的浮点张量，像素缩放到 [0, 1]。
    transform = ToTensor()

    # train=True 读取官方训练集。
    # download=True 在数据缺失时下载，已存在时直接使用。
    full_train_dataset = FashionMNIST(
        root=str(data_dir),
        train=True,
        transform=transform,
        download=True,
    )

    # 验证集必须非空，同时需要给训练集保留样本。
    if not 0 < config.val_size < len(full_train_dataset):
        raise ValueError("val_size 必须大于 0 且小于训练集总样本数")

    # 根据配置计算训练集大小，而不是在这里写死 54000。
    train_size = len(full_train_dataset) - config.val_size

    # manual_seed 固定这个生成器的随机种子，使划分可以重复。
    split_generator = torch.Generator().manual_seed(config.seed)

    # random_split 返回两个子集；左侧两个变量分别接收它们。
    # 两个子集使用不同的样本索引，不会互相重叠。
    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_size, config.val_size],
        generator=split_generator,
    )

    # train=False 读取官方测试集，留给最后的独立评估。
    test_dataset = FashionMNIST(
        root=str(data_dir),
        train=False,
        transform=transform,
        download=True,
    )

    # shuffle=True 在每次遍历训练集时打乱样本顺序。
    # num_workers=0 表示由当前进程加载数据，先采用简单的运行方式。
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
    )

    # 验证与测试不更新参数，这里按固定顺序读取。
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # 按训练、验证、测试的顺序返回，调用处也按这个顺序接收。
    return train_loader, val_loader, test_loader