# FashionMNIST 服装图像分类器

这是一个使用 PyTorch 实现的 FashionMNIST 图像分类项目。项目包含配置读取、数据划分、模型定义、训练、验证、最佳模型保存、日志记录和测试评估等完整流程。

本项目的主要目的不是追求最高准确率，而是学习一个结构清晰、可以重复运行的 PyTorch 分类项目如何组织。

## 当前结果

使用 RTX 5060 Laptop GPU 训练5轮后的结果：

| 指标 | 结果 |
| --- | ---: |
| 最佳模型轮次 | Epoch 5 |
| 最佳验证准确率 | 88.00% |
| 测试损失 | 0.3533 |
| 测试准确率 | 87.45% |

训练集、验证集和测试集分别用于：

- 训练集：更新模型参数；
- 验证集：比较每轮表现并选择最佳模型；
- 测试集：对保存的最佳模型进行最终评估。

## 项目结构

```text
fashion-mnist-classifier/
├── checkpoints/             # 保存训练过程中表现最好的模型
│   └── best_model.pt
├── configs/                 # 保存可调整的训练配置
│   └── default.yaml
├── data/                    # FashionMNIST 数据集
├── logs/                    # 训练日志
│   └── train.log
├── config.py                # 读取 YAML 并创建训练配置对象
├── dataset.py               # 下载、划分并加载数据集
├── eval.py                  # 加载最佳模型并在测试集上评估
├── model.py                 # 定义神经网络结构
├── train.py                 # 组织训练和验证流程
├── utils.py                 # 保存检查点和创建日志记录器
└── README.md                # 项目说明
```

## 模型结构

输入是大小为 `28 × 28` 的单通道灰度图片。

```text
[batch_size, 1, 28, 28]
          ↓
Flatten
          ↓
[batch_size, 784]
          ↓
Linear(784, 128)
          ↓
ReLU
          ↓
Linear(128, 10)
          ↓
[batch_size, 10]
```

最后的10个输出值是图片属于10个服装类别的分数（logits）。程序使用 `argmax(dim=1)` 选出分数最高的类别编号。

## 数据划分

项目使用 FashionMNIST 数据集：

| 数据 | 数量 | 用途 |
| --- | ---: | --- |
| 训练集 | 55,000 | 更新模型参数 |
| 验证集 | 5,000 | 选择最佳模型 |
| 测试集 | 10,000 | 最终评估 |

训练集和验证集来自官方60,000张训练图片。项目使用固定随机种子 `42` 进行划分，使每次运行得到相同的数据划分。

## 运行环境

本次实验使用的主要环境：

```text
Python 3.13.9
PyTorch 2.14.0+cu130
torchvision 0.29.0+cu130
PyYAML 6.0.3
CUDA 13.0
```

项目也可以使用 CPU 运行，但训练速度通常比 GPU 慢。

## 创建虚拟环境

在项目目录中创建虚拟环境：

```powershell
python -m venv .venv
```

在 PowerShell 中激活：

```powershell
.\.venv\Scripts\Activate.ps1
```

安装项目依赖：

```powershell
python -m pip install torch torchvision pyyaml
```

如果需要安装支持特定 CUDA 版本的 PyTorch，应以 PyTorch 官方安装页面给出的命令为准。

## 训练配置

训练参数保存在 `configs/default.yaml`：

```yaml
batch_size: 64
learning_rate: 0.001
epochs: 5
device: auto
```

`device: auto` 表示：

- CUDA 可用时使用 GPU；
- CUDA 不可用时使用 CPU。

## 开始训练

首先进入项目目录：

```powershell
cd D:\code\26fall_AiLearning\projects\fashion-mnist-classifier
```

然后运行：

```powershell
python train.py
```

训练流程为：

```text
读取配置
→ 创建 DataLoader
→ 创建模型
→ 前向传播
→ 计算损失
→ 反向传播
→ 更新参数
→ 在验证集上评估
→ 保存验证准确率最高的模型
```

训练信息会同时显示在终端中，并追加写入：

```text
logs/train.log
```

在 PowerShell 中按 UTF-8 编码查看日志：

```powershell
Get-Content -Encoding UTF8 .\logs\train.log
```

## 测试最佳模型

训练完成后运行：

```powershell
python eval.py
```

`eval.py` 会：

1. 创建与训练阶段结构相同的模型；
2. 读取 `checkpoints/best_model.pt`；
3. 加载 `model_state_dict`；
4. 在官方测试集上计算损失和准确率。

## 检查点内容

`checkpoints/best_model.pt` 保存了：

- 训练轮次；
- 模型参数 `model_state_dict`；
- 优化器状态 `optimizer_state_dict`；
- 验证损失；
- 验证准确率。

准确率只是一个评估数字，不能恢复模型。真正让模型保留训练结果的是 `model_state_dict` 中的权重和偏置。

## 本项目练习的知识

- Python 函数、类、继承、类型标注和模块导入；
- `dataclass`、YAML、`pathlib` 和异常处理；
- Tensor、shape、device 和 DataLoader；
- `nn.Module`、损失函数和优化器；
- 训练模式、评估模式和关闭梯度；
- 保存和加载模型检查点；
- 使用 logging 同时记录终端信息和日志文件；
- 使用 Git 保存项目开发过程。

## 后续改进方向

- 使用卷积神经网络（CNN）提高准确率；
- 增加数据标准化；
- 加入学习率调度器；
- 增加随机种子设置，提高完整训练过程的可复现性；
- 绘制训练损失和准确率曲线；
- 添加针对配置、数据和模型输出形状的自动化测试。
