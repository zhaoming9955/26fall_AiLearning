""" 定义神经网络 """

# 导入 PyTorch，用于创建和操作 Tensor。
import torch

# nn 提供构建神经网络所需的类。
from torch import nn


class FashionClassifier(nn.Module):
    """用于识别 FashionMNIST 服装类别的神经网络。"""

    def __init__(self):
        """定义模型包含的神经网络层。"""

        # 初始化父类 nn.Module。
        # 前后都是两个下划线：__init__
        super().__init__()

        # 数据会按照顺序经过这些网络层。
        self.network = nn.Sequential(
            # [batch_size, 1, 28, 28]
            # 转换成 [batch_size, 784]。
            nn.Flatten(),

            # 把784个输入特征转换成128个隐藏特征。
            nn.Linear(28 * 28, 128),

            # 添加非线性能力。
            nn.ReLU(),

            # 为每张图片输出10个类别分数。
            nn.Linear(128, 10),
        )

    # forward 与 __init__ 同级，都属于 FashionClassifier。
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """规定图片进入模型后的计算流程。"""

        logits = self.network(images)

        return logits


# 下面的代码位于类的外面，只在直接运行文件时执行。
if __name__ == "__main__":
    # 创建模型实例。
    model = FashionClassifier()

    # 创建64张用于检查模型的假图片。
    fake_images = torch.randn(64, 1, 28, 28)

    # model(...) 会自动调用 model.forward(...)。
    logits = model(fake_images)

    print(model)
    print(f"输入形状：{fake_images.shape}")
    print(f"输出形状：{logits.shape}")