""" 评估训练好的模型:加载模型, 计算准确率 """

"""
checkpoint 可以理解成一次训练的“存档包”，
里面不仅有模型参数，还有保存轮次、验证准确率、优化器状态等信息。
"""

from pathlib import Path

import torch
from torch import nn

from config import load_config
from dataset import create_dataloaders
from model import FashionClassifier
from train import choose_device, evaluate

def main(): 
    """加载最佳模型, 并计算测试集损失和准确率"""
    # 读取YAML中保存的训练配置
    config = load_config(Path("configs/default.yaml"))
    
    # 自动选择 CUDA 或 CPU
    device = choose_device(config.device)
    print(f"实际运行设备: {device}")
    
    # 创建数据加载器
    # 这里只使用测试集, 因此前两个返回值用下划线忽略
    _, _, test_loader = create_dataloaders(batch_size=config.batch_size)
    
    
    # 创建与训练时结构相同的模型
    model = FashionClassifier().to(device)
    
    # 最佳模型文件的位置
    checkpoint_path = Path("checkpoints/best_model.pt")
    
    # 若模型文件不存, 就主动给出容易理解的错误
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {checkpoint_path}")
    
    # 从硬盘读取检查点
    # map_location 保证数据被加载到当前使用的设备
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    
    # 将训练好的参数装入模型
    model.load_state_dict(checkpoint["model_state_dict"])
    
    # 创建与训练阶段相同的损失函数
    loss_function = nn.CrossEntropyLoss()
    
    # 使用测试集进行最终评估
    test_loss, test_accuracy = evaluate(
        model = model,
        data_loader = test_loader,
        loss_function = loss_function,
        device = device,
    )
    
    # 显示检查点信息和最终测试结果。
    print(f"加载的模型来自第 {checkpoint['epoch']} 轮")
    print(
        f"保存时验证准确率："
        f"{checkpoint['validation_accuracy']:.2%}"
    )
    print(f"测试损失：{test_loss:.4f}")
    print(f"测试准确率：{test_accuracy:.2%}")


# 只有直接运行 eval.py 时才调用 main()。
if __name__ == "__main__":
    main()