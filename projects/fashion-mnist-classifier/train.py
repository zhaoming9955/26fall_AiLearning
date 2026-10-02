""" 启动并组织训练 """
from utils import save_checkpoint, setup_logger
from pathlib import Path
import torch
from torch import nn
from config import TrainConfig, load_config
from dataset import create_dataloaders
from model import FashionClassifier
from torch.utils.data import DataLoader

def choose_device(device_name: str) -> torch.device:
    """根据配置选择CPU或GPU设备。"""
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"    
    return torch.device(device_name)

def describe_config(config: TrainConfig) -> str:
    """把训练配置转换为便于阅读的字符串"""
    return(
        f"batch_size={config.batch_size}, "
        f"learning_rate={config.learning_rate}, "
        f"epochs={config.epochs}, "
        f"device={config.device}"
    )
    
def train_one_epoch(
    model : nn.Module,
    data_loader : DataLoader,
    loss_function : nn.Module,
    optimizer : torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """使用全部训练数据训练一轮模型, 并返回平均损失和准确率"""
    model.train()  # 切换到训练模式
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    # 每次循环取出一个批次的图片和标签
    for images, labels in data_loader:
        # 把当前批次移动到模型所在设备
        images = images.to(device)
        labels = labels.to(device)
        
        # 清除上一个批次留下的梯度
        optimizer.zero_grad()
        
        # 前向传播, 获得每个类别的分数
        logits = model(images)
        
        # 计算当前批次的损失
        loss = loss_function(logits, labels)
        
        # 反向传播, 计算模型参数的梯度
        loss.backward()
        
        # 根据梯度更新模型参数
        optimizer.step()
        
        # 当前批次实际包含的样本数量
        batch_size = images.size(0)
        
        # loss.item() 是当前批次的平均损失。
        # 乘以批次大小后累加，最后再计算全体平均值。
        total_loss += loss.item() * batch_size

        # 在10个类别分数中找到最大值所在的位置。
        predictions = logits.argmax(dim=1)

        # 统计预测正确的图片数量。
        correct_predictions += (
            predictions == labels
        ).sum().item()

        total_samples += batch_size

    average_loss = total_loss / total_samples
    accuracy = correct_predictions / total_samples

    return average_loss, accuracy

@torch.no_grad()
def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """在验证集上计算平均损失和准确率"""
    
    # 切换到评估模式
    model.eval()
    total_loss = 0.0
    correct_prediction = 0
    total_samples = 0
    
    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        # 验证阶段只进行前向传播
        logits = model(images)
        loss = loss_function(logits, labels)
        
        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        
        predictions = logits.argmax(dim=1)
        correct_prediction += (
            predictions == labels
        ).sum().item()
        
        total_samples += batch_size
        
    average_loss = total_loss / total_samples
    accuracy = correct_prediction / total_samples
    return average_loss, accuracy
        
def main() -> None:
    """加载配置, 数据和模型. 并完成一次参数更新"""
    
    # 创建日志记录器，将运行信息同时输出到终端和文件。
    logger = setup_logger(Path("logs/train.log"))
    
    # 读取 YAML 配置, 并转换成TrainConfig对象
    config_path = Path("configs/default.yaml")
    config = load_config(config_path)
    
    logger.info(
        "训练配置加载成功：%s",
        describe_config(config),
    )
    
    # 根据配置选择CPU或GPU
    device = choose_device(config.device)
    logger.info("实际运行设备：%s", device)
    
    # 创建训练, 验证和测试数据加载器
    train_loader, validation_loader, test_loader = create_dataloaders(
        batch_size = config.batch_size
    )
    
    logger.info("训练批次数量：%d", len(train_loader))
    logger.info("验证批次数量：%d", len(validation_loader))
    logger.info("测试批次数量：%d", len(test_loader))
    
    # 创建模型, 并将模型参数移动到目标设备
    model = FashionClassifier().to(device)
    
    # CrossEntropyLoss 用于多类别分类
    # 它会比较模型输出的 logits 和真实标签
    loss_function = nn.CrossEntropyLoss()
    
    # Adam优化器根据梯度更新模型参数
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr = config.learning_rate,
    )
    
    # 记录目前最高的验证准确率。
    best_validation_accuracy = 0.0

    # 注意拼写是 checkpoint，不是 checkpoint。
    checkpoint_path = Path("checkpoints/best_model.pt")

    # 完整训练 config.epochs 轮。
    for epoch in range(config.epochs):
        train_loss, train_accuracy = train_one_epoch(
            model=model,
            data_loader=train_loader,
            loss_function=loss_function,
            optimizer=optimizer,
            device=device,
        )

        validation_loss, validation_accuracy = evaluate(
            model=model,
            data_loader=validation_loader,
            loss_function=loss_function,
            device=device,
        )

        # 输出当前轮次的训练和验证结果。
        logger.info(
        "Epoch %d/%d | "
        "训练损失：%.4f | "
        "训练准确率：%.2f%% | "
        "验证损失：%.4f | "
        "验证准确率：%.2f%%",
        epoch + 1,
        config.epochs,
        train_loss,
        train_accuracy * 100,
        validation_loss,
        validation_accuracy * 100,
        )

        # 验证准确率提升时，保存新的最佳模型。
        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                validation_loss=validation_loss,
                validation_accuracy=validation_accuracy,
                path=checkpoint_path,
            )

            logger.info(
                "已保存最佳模型：验证准确率=%.2f%%",
                validation_accuracy * 100,
            )

    
# 只有直接运行 train.py 时才调用 main()。
if __name__ == "__main__":
    main()