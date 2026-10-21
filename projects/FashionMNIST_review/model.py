""" 
定义 FashionMNIST 的全连接分类模型; 将输入图像转换为各类别的原始分数, 供后续计算分类损失
"""

# Tensor用于标注张量类型; nn提供神经网络的基类和常用层
from torch import Tensor, nn

class SimpleClassifier(nn.Module):
    def __init__(self, input_size: int, num_classes: int):
        """创建网络; 两个参数分别表示输入像素和类别数"""
        
        # 先初始化父类nn.Module, 再定义自己的网络层
        # super()用于访问父类提供的方法
        super().__init__()
        
        # 保存模型信息, 注意: 赋值末尾不能加逗号, 否则整数784会变成元组(784,)
        self.input_size = input_size
        self.num_classes = num_classes
        
        # 保留第0维的批次大小, 从第一维度开始展平.
        # [批次大小, 1, 28, 28] -> [批次大小, 784]
        self.flatten = nn.Flatten(start_dim=1)
        
        # Sequential 按照书写顺序, 依次执行内部的网络层
        self.layers = nn.Sequential(
            
            # 全连接层: 将每张图的 input_size 个像素映射为 128 个特征
            nn.Linear(input_size, 128),
            
            # 激活函数: 把负数变为0, 保留非负数, 引入非线性
            nn.ReLU(),
            
            # 分类层: 将128个特征映射为 num_classes 个类别分数
            nn.Linear(128, num_classes)
        )
        
    def forward(self, images: Tensor) -> Tensor:
        """定义图像经过网络的计算过程, 返回每张图的类别分数"""
        
        # 第一步: 将每张二维图像展平成一维的像素向量
        features = self.flatten(images)
        
        # 第二步: 让向量依次经过全连接层、激活函数和分类层
        logits = self.layers(features)
        
        # logits是原始分数, 不是概率值; 后续使用CrossEntropy时, 直接传入这些分数即可
        return logits
        
        
    def describe(self) -> str:
        """返回模型信息, 保留之前练习过的说明功能"""
        
        return (
            f"Classifier: input_size = {self.input_size}, "
            f"num_classes = {self.num_classes}"
        )