from dataclasses import dataclass
from pathlib import Path

import yaml

"""这部分规定了训练配置的结构，包括批量大小、学习率、训练轮数和运行设备。"""
@dataclass
class TrainConfig:
    batch_size: int
    learning_rate: float
    epochs: int
    device: str

def load_config(path:Path) -> TrainConfig:  # ->TrainConfig 表示这个函数的返回类型是TrainConfig的实例
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}") # raise:主动抛出异常,中断程序执行
    """读取文件内容"""
    contents = path.read_text(encoding="utf-8")

    """把YAML字符串解析成字典"""
    data = yaml.safe_load(contents)

    """这个部件的核心目的是将YAML中松散的字典数据转换成一个结构清晰的TrainConfig对象"""
    config = TrainConfig(
        batch_size = data["batch_size"],
        learning_rate = data["learning_rate"],
        epochs = data["epochs"],
        device = data["device"]
    )
    return config
