from pathlib import Path
from config import TrainConfig, load_config

def describe_config(config: TrainConfig) -> str:
    return (
        f"batch_size={config.batch_size}, "
        f"learning_rate={config.learning_rate}, "
        f"epochs={config.epochs}, "
        f"device={config.device}"
    )
    
def main() -> None:
    config_path = Path("configs/default.yaml")
    config = load_config(config_path)
    
    print("训练配置加载成功: ")
    print(describe_config(config))
    
if __name__ == "__main__":
    main()