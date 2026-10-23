"""
组织模型训练与验证，保存最佳模型，
并将运行配置、每轮指标和保存信息记录到终端与日志文件。
"""

# 导入项目内部的配置模块和数据加载模块。
import torch


from torch import nn, optim
from config import TrainConfig, describe_config
from dataset import build_dataloaders
from model import SimpleClassifier
from torch.utils.data import DataLoader
from eval import evaluate
from dataclasses import asdict
from pathlib import Path
from utils import setup_logger                  # 导入日志配置函数。

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: str,
) -> tuple[float, float]:
    """完整遍历一次训练集，更新参数并汇总训练指标"""

    # 设置训练模式, 并初始化本轮统计量
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    # 每次循环从加载器取出一批图像及其标签
    for images, labels in loader:
        # 将当前批次移动到与模型相同的设备
        images = images.to(device)
        labels = labels.to(device)
        
        # 清除上一批次的梯度, 随后计算本批次的预测与损失
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        
        # 根据损失梯度, 再由优化器更新参数
        loss.backward()
        optimizer.step()
        
        # size(0) 读取第0维, 即当前批次实际包含的样本数.
        batch_size = images.size(0)
        
        # loss 默认是批次平均值, 乘样本数得到批次损失总和
        # 最后除以总样本数, 得到按样本加权的整轮平均损失
        total_loss += loss.item() * batch_size
        
        # argmax(dim=1) 对每张图的10个分数取最大值所在的类别编号
        predictions = logits.argmax(dim=1)
        
        # 逐个比较预测和标签
        total_correct += (predictions == labels).sum().item()
        total_samples += batch_size
        
    # 准确率返回0~1的比例, 输出时再格式化为百分比
    average_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return average_loss, accuracy

def main() -> None:
    """加载数据，并输出检查信息。"""

    # 创建训练配置。
    config = TrainConfig()

    # 通过脚本路径确定项目位置，并创建日志记录器。
    project_dir = Path(__file__).resolve().parent
    logger = setup_logger(project_dir / "logs")

    # 标记一次新的运行，方便区分追加文件中的不同实验。
    logger.info("=" * 50)
    logger.info("开始新的训练")
    logger.info(describe_config(config))

    # 元组解包：依次接收函数返回的三个数据加载器。
    train_loader, val_loader, test_loader = build_dataloaders(config)

    # loader.dataset 是加载器使用的数据集。
    # len(loader.dataset) 表示样本数量；len(loader) 表示批次数量。
    # print(f"训练集样本数: {len(train_loader.dataset)}")
    # print(f"验证集样本数: {len(val_loader.dataset)}")
    # print(f"测试集样本数: {len(test_loader.dataset)}")
    
    # 同时记录到终端和日志文件。
    logger.info(f"训练集样本数: {len(train_loader.dataset)}")
    logger.info(f"验证集样本数: {len(val_loader.dataset)}")
    logger.info(f"测试集样本数: {len(test_loader.dataset)}")

    # 模型、损失函数和优化器在多轮训练开始前创建一次， 后续各轮
    # 后续各轮延续已有的模型参数与优化器状态
    model = SimpleClassifier(input_size=28*28, num_classes=10)
    model = model.to(config.device)
    
    criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # 记录模型结构说明。
    logger.info(model.describe())
    

    checkpoint_dir = project_dir / "checkpoints"
    
    # parents = True 允许创建缺失的父目录
    # exist_ok = True 表示目录已经存在时不报错
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # 每次验证准确率提高时, 覆盖这个文件
    checkpoint_path = checkpoint_dir / "best.pt"
    
    # 准确率范围是0~1, 初始化为-1, 确保第一轮也能保存
    best_val_acc = -1.0
    best_epoch = 0
    
    
    # 每轮先更新模型参数，再用验证集评估当前模型。
    for epoch in range(1, config.epochs + 1):
        # 训练函数内部调用 model.train()，切回训练模式。
        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=config.device,
        )

        # 使用独立的验证集计算指标，整个验证过程不更新参数。
        val_loss, val_acc = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=config.device,
        )

        # # 同时输出训练与验证结果，便于观察它们随轮次的变化。
        # print(
        #     f"Epoch {epoch}/{config.epochs} | "
        #     f"train_loss={train_loss:.4f} | "
        #     f"train_acc={train_acc:.2%} | "
        #     f"val_loss={val_loss:.4f} | "
        #     f"val_acc={val_acc:.2%}"
        # )

        # 保留原来的指标格式，改由日志记录器输出。
        logger.info(
            f"Epoch {epoch}/{config.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"train_acc={train_acc:.2%} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.2%}"
        )
        
        # 只根据验证集准确率选择最佳模型; 相同时保持较早的一轮
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            
            # 将恢复模型所需的状态和实验信息组织为字典
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "config": asdict(config),
            }
            
            # 将当前状态立即写入磁盘, 保留这一轮的参数
            torch.save(checkpoint, checkpoint_path)
            
            # 记录本次最佳模型的保存信息。
            logger.info(
                f"已保存最佳模型：epoch={epoch} | "
                f"val_acc={val_acc:.2%}"
            )

    # 记录最终最佳结果和模型位置。
    logger.info(
        f"训练结束，最佳轮次={best_epoch}，"
        f"最佳验证准确率={best_val_acc:.2%}"
    )
    logger.info(f"模型保存位置：{checkpoint_path}")
        
        

# 直接运行此文件时执行入口函数。
if __name__ == "__main__":
    main()
    """组织训练与验证，记录日志，并保存验证准确率最高的模型。"""