"""存放项目中通用的辅助函数。"""

import logging
from pathlib import Path

import torch
from torch import nn


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    validation_loss: float,
    validation_accuracy: float,
    path: Path,
) -> None:
    """保存模型、优化器以及当前训练信息。"""

    # 如果检查点目录不存在，就自动创建。
    path.parent.mkdir(parents=True, exist_ok=True)

    # 将恢复模型或训练时需要的信息整理成字典。
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "validation_loss": validation_loss,
        "validation_accuracy": validation_accuracy,
    }

    # 将检查点字典保存到指定文件。
    torch.save(checkpoint, path)


def setup_logger(log_path: Path) -> logging.Logger:
    """创建同时向终端和文件输出信息的日志记录器。"""

    # 如果日志目录不存在，就自动创建。
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # 获取本项目使用的日志记录器。
    logger = logging.getLogger("fashion_mnist")

    # INFO级别用于记录程序正常运行时的重要信息。
    logger.setLevel(logging.INFO)

    # 清除之前添加的处理器，避免日志被重复输出。
    logger.handlers.clear()

    # 阻止日志继续传递给根记录器，避免重复输出。
    logger.propagate = False

    # 规定每条日志的格式：时间、级别和具体信息。
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    # 创建终端处理器，将日志显示在终端中。
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # 创建文件处理器，将日志写入指定文件。
    file_handler = logging.FileHandler(
        log_path,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    # 将终端处理器和文件处理器添加到日志记录器中。
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger