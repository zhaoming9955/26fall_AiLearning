""" 
提供可复用的评估函数。
被训练程序导入时用于验证；直接运行时加载最佳模型并评估测试集。
"""



import torch

# Path 用于定位项目与 checkpoint 文件。
from pathlib import Path

# 根据保存的配置恢复数据加载方式，并重建模型。
from config import TrainConfig
from dataset import build_dataloaders
from model import SimpleClassifier
from torch import nn
from torch.utils.data import DataLoader

# 在函数执行期间关闭梯度记录, 减少评估的内存开销
@torch.no_grad()

def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> tuple[float, float]:
    """遍历评估数据, 返回平均损失与准确率, 不更新模型参数"""

    model.eval()
    
    # 初始化整个评估数据集的统计量
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    # 逐批读取图像与标签
    for images, labels in loader:
        # 将数据移动到模型所在设备
        images = images.to(device)
        labels = labels.to(device)
        
        # 使用当前模型计算预测与损失
        logits = model(images)
        loss = criterion(logits, labels)
        
        # 按实际样本数据累计损失, 兼顾最后一个不满批次
        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        
        # 取最大分数对应的类别编号, 再统计正确预测的数量
        predictions = logits.argmax(dim=1)
        
        # 累计当前批次的正确预测数量与实际样本数
        total_correct += (predictions == labels).sum().item()
        total_samples += batch_size
         
    # 循环结束后, 按整个评估集的样本数计算平均指标
    average_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return average_loss, accuracy

def main() -> None:
    """从磁盘恢复最佳模型，独立计算测试集指标。"""

    # 根据脚本位置定位模型文件，避免依赖终端所在目录。
    project_dir = Path(__file__).resolve().parent
    checkpoint_path = project_dir / "checkpoints" / "best.pt"

    # map_location="cpu" 将保存的张量加载到 CPU。
    # weights_only=True 使用适合张量和基础数据类型的受限加载方式。
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    # ** 将字典展开为关键字参数，例如 batch_size=64。
    # 使用训练时保存的配置，并明确在 CPU 上执行独立评估。
    config = TrainConfig(**checkpoint["config"])
    config.device = "cpu"

    # 先重建与训练时一致的网络结构，再载入训练好的参数。
    model = SimpleClassifier(input_size=28 * 28, num_classes=10)
    model = model.to(config.device)

    # strict=True 要求保存的参数名称与当前模型完全匹配。
    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    # 复用数据模块，只使用返回的第三个加载器，即测试集。
    # 下划线表示这里不使用对应的训练和验证加载器。
    _, _, test_loader = build_dataloaders(config)
    criterion = nn.CrossEntropyLoss()

    # 复用评估函数；其中已有评估模式与关闭梯度记录的设置。
    test_loss, test_acc = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=config.device,
    )

    # 输出模型来源及最终测试结果。
    print(f"模型文件：{checkpoint_path}")
    print(f"保存轮次：{checkpoint['epoch']}")
    print(f"保存的验证准确率：{checkpoint['val_acc']:.2%}")
    print(f"测试集样本数：{len(test_loader.dataset)}")
    print(f"测试损失：{test_loss:.4f}")
    print(f"测试准确率：{test_acc:.2%}")


# 直接运行 eval.py 才执行独立测试。
# train.py 导入 evaluate 时，不会执行这里的 main()。
if __name__ == "__main__":
    main()