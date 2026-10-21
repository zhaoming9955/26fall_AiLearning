"""
组织训练程序的执行流程。当前阶段加载配置与数据，检查数据集大小和一个批次的形状。
"""

# 导入项目内部的配置模块和数据加载模块。
from config import TrainConfig, describe_config
from dataset import build_dataloaders


def main() -> None:
    """加载数据，并输出检查信息。"""

    # 创建配置，沿用已有的参数摘要输出。
    config = TrainConfig()
    print(describe_config(config))

    # 元组解包：依次接收函数返回的三个数据加载器。
    train_loader, val_loader, test_loader = build_dataloaders(config)

    # loader.dataset 是加载器使用的数据集。
    # len(loader.dataset) 表示样本数量；len(loader) 表示批次数量。
    print(f"训练集样本数: {len(train_loader.dataset)}")
    print(f"验证集样本数: {len(val_loader.dataset)}")
    print(f"测试集样本数: {len(test_loader.dataset)}")

    # iter() 创建迭代器，next() 从中取出一个批次。
    # 这个批次包含图像张量和标签张量，用两个变量分别接收。
    images, labels = next(iter(train_loader))

    # shape 是张量各维度的大小；dtype 是张量中元素的数据类型。
    print(f"图像形状: {images.shape}")
    print(f"标签形状: {labels.shape}")
    print(f"图像类型: {images.dtype}")
    print(f"标签类型: {labels.dtype}")


# 直接运行此文件时执行入口函数。
if __name__ == "__main__":
    main()