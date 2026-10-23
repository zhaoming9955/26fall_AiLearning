""" 
集中定义训练参数，并提供配置摘要函数，
供数据加载、训练和评估模块使用。
"""

from dataclasses import dataclass


@dataclass    # @dataclass 会自动帮我们生成初始化方法
class TrainConfig:
    batch_size: int = 64
    learning_rate: float = 0.001
    epochs: int = 5
    device: str = "cpu"
    data_dir: str = "data"      # 数据保存目录；后续代码会将它解析为相对于项目目录的路径。
    val_size: int = 6000        # 从官方训练集中取出 6000 张，作为验证集。
    seed: int = 42              # 固定随机划分使用的种子，方便重复获得相同的数据划分。


def describe_config(config: TrainConfig) -> str:
    return(
        f"batch_size = {config.batch_size}\n"
        f"learning_rate = {config.learning_rate}\n"
        f"epochs = {config.epochs}\n"
        f"device = {config.device}"
    )


