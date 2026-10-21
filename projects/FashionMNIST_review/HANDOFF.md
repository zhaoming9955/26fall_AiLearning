<!-- 文件作用：记录 FashionMNIST_review 的学习目标、教学约定、当前代码状态、下载问题和后续接续步骤，供新聊天读取后继续指导。 -->

# FashionMNIST_review 聊天交接

更新日期：2026-10-21（Asia/Shanghai）。以下状态以本次交接前读取的本地文件、环境检查和用户反馈为准；后续聊天开始时应检查是否有新变化。

## 最新接续记录：本机环境和数据验收已通过

2026-10-21 本节为最新状态；后文关于旧电脑环境和数据缺失的说明保留为历史记录，接续时以本节和实际文件为准。

- 当前项目路径：`D:\CODE\26Fall\26fall_AiLearning\projects\FashionMNIST_review`。
- 用户最新明确要求：所有 Python 代码仅由助手在聊天中提供带注释版本，用户自己抄写，禁止助手直接创建或修改项目 Python 文件。终端命令仍由助手执行。用户已要求撤回此前代写的全部 Python 改动，撤回操作已完成。
- 注释要求：每个文件头说明功能，每个功能模块有简要说明，陌生语法和接口就近备注；每完成一个或多个文件，向用户对照任务清单报告完成与未完成项。
- 已在 `D:\Anaconda_Folder\envs\cqu-ai\python.exe` 对应的环境安装项目依赖：Python 3.11.17、torch 2.11.0+cpu、torchvision 0.26.0+cpu、Pillow 12.3.0。
- 实际通过 torch、torchvision、Pillow 导入、张量矩阵乘法和 `python -m pip check` 检查。
- PowerShell 默认 `python` 仍指向 base。base 的 torch/torchvision 导入可复现 Pillow DLL 报错；本次通过使用项目独立环境解决项目运行问题，没有修复或卸载 base 中的包。
- 助手执行时使用 `conda run --no-capture-output -n cqu-ai python train.py`，无需依赖当前终端的激活状态。
- 已实际运行该入口，退出码为 0：四份数据下载完成，训练/验证/测试样本数为 54000/6000/10000，图像形状为 `[64, 1, 28, 28]`、标签形状为 `[64]`，类型分别为 float32/int64。
- 此前助手直接实现模型、修改入口与配置、创建测试，违背了用户希望自己抄写代码的教学方式；已将 `config.py`、`model.py`、`train.py` 恢复到代写前的版本，并移除助手新增的 `tests/test_model.py`。原有其他 Python 文件、依赖环境和数据保留。
- 撤回后，用户已自行抄写完成 `model.py`，并自行修正了 Sequential 中缺少逗号及 forward 中展平调用错误。当前模型继承 `nn.Module`，使用 `Flatten(start_dim=1)`、`Linear(input_size, 128)`、`ReLU()`、`Linear(128, num_classes)`，保留 `describe()`，输入尺寸已正确保存为整数。
- 助手已对用户保存的模型执行真实训练批次前向验收：输入 `[64, 1, 28, 28]`，输出 `[64, 10]`、float32，所有分数有限；单张图像输出 `[1, 10]` 也通过检查。模型尚未训练，尚无准确率结果。
- 当前 `train.py` 仍只检查配置和数据，尚未接入模型；`config.py` 文件头说明仍待补齐。聊天中已提供 `train.py` 的模型导入、创建、设备移动和前向检查代码，等待用户自行抄写保存。
- 已完成：项目环境、配置运行、数据加载验收、用户自行编写的模型定义与前向传播。尚未完成：配置文件头说明、入口接入模型、单批次参数更新、训练循环、验证指标、checkpoint、日志、独立评估、最终结果说明。
- 下一步：检查用户保存的 `train.py` 是否已接入模型，运行入口验收后再指导单批次训练。禁止因用户说“继续推进”而直接写入 Python 文件。

## 1. 用户目标与教学约定

用户正在重启 Week 1 学习，当前只推进 FashionMNIST 分类项目复现。

- 周计划目录：`D:\code\26fall_AiLearning\weekly\week01`
- 项目工作区：`D:\code\26fall_AiLearning\projects\FashionMNIST_review`
- 周计划原目标包括 train、validation、checkpoint、logging，最终能够保存最佳模型并独立评估。
- 用户当前要求只围绕项目推进，不扩展到算法题、求职、Linux、SSH 等周计划中的其他内容。
- 用户原话：**“每个新文件需要在文件头写好注释解释文件作用，每段代码的每个模块给出简要注释，比较少见的接口或语法也要给出功能描述。”**
- 当前协作方式是逐步教学：助手查看现有文件，给出带注释的代码与运行命令，用户自己写入、运行，再反馈结果。保持这种方式；用户尚未要求助手直接代写全部项目。
- 一次完成一个可以运行、可以检查输出的小步骤。解释应贴近当前代码；出现报错时先解决当前问题，再继续后面的模型与训练。
- 不以代码行数为目标，也没有约定必须复现某篇论文、特定网络或特定准确率。

## 2. 已确认的运行环境

交接前实际调用目标解释器，确认以下版本能够导入：

| 项目 | 状态 |
|---|---|
| 操作系统 / 终端 | Windows / PowerShell |
| Conda 环境 | `cqu-ai` |
| Python 路径 | `D:\Anaconda\envs\cqu-ai\python.exe` |
| Python 版本 | `3.11.16` |
| torch | `2.11.0+cu128` |
| torchvision | `0.26.0+cu128` |
| 当前项目配置设备 | `cpu` |

早先用户在 `(base)` 中运行，Python 为 `D:\Anaconda\python.exe`，后来已切换到 `(cqu-ai)`。助手自身终端默认解释器可能不同，检查项目时应明确使用上面的目标解释器。

CUDA 构建版本不等于已验证 GPU 可用；当前任务不需要切换 GPU。

```powershell
# 明确激活项目学习环境。
conda activate D:\Anaconda\envs\cqu-ai

# 进入项目后运行入口。
Set-Location 'D:\code\26fall_AiLearning\projects\FashionMNIST_review'
python train.py
```

## 3. 当前代码状态

| 文件 / 目录 | 当前状态 |
|---|---|
| `config.py` | 已有 `TrainConfig` 和 `describe_config(config)`；配置输出已由用户运行确认。尚无文件头用途说明，后续修改时补齐。 |
| `dataset.py` | 已写入带文件头和模块注释的数据读取、划分、DataLoader 代码；尚未完成一次成功的数据加载验收。 |
| `train.py` | 已写入带文件头和模块注释的入口，目前仅输出配置、数据集大小和一个批次的形状及类型；尚无训练循环。 |
| `model.py` | 原有普通 Python 类 `SimpleClassifier`，尚未继承 `nn.Module`，不具备前向传播和可训练层；存在下面记录的逗号问题。 |
| `eval.py`、`utils.py` | 空文件。 |
| `README.md`、`requirements.txt` | 已在 Git 同步准备阶段补充：README 记录跨电脑克隆、安装、数据下载恢复和当前进度；requirements 固定 torch 2.11.0 与 torchvision 0.26.0。 |
| `.gitignore`、`HANDOFF.md` | 已在 Git 同步准备阶段加入：排除本地数据与运行产物，并保留本交接说明。 |
| `configs/default.yaml` | 空文件；目前配置实际来自 `config.py`，不需要立即引入 YAML。 |
| `test_1.ipynb` | 用户的学习笔记与练习，不作为当前主程序入口；内容可能包含旧示例，判断实现以 `.py` 文件为准。 |
| `checkpoints/`、`logs/`、`outputs/` | 项目中已有这些目录；目前尚无已验证的训练结果。 |

### 配置字段

```python
# 当前 TrainConfig 的字段与默认值；此处是状态记录，无需重复创建类。
batch_size: int = 64
learning_rate: float = 0.001
epochs: int = 5
device: str = "cpu"
data_dir: str = "data"
val_size: int = 6000
seed: int = 42
```

`describe_config(config: TrainConfig) -> str` 通过 `return` 返回四行字符串，目前只列出 batch_size、learning_rate、epochs、device。用户已成功得到：

```text
batch_size = 64
learning_rate = 0.001
epochs = 5
device = cpu
```

### 数据模块接口及行为

`build_dataloaders(config: TrainConfig) -> tuple[DataLoader, DataLoader, DataLoader]`

- 返回顺序固定为训练、验证、测试。
- 数据目录通过 `Path(__file__).resolve().parent / config.data_dir` 计算，默认位于项目下的 `data/`。
- 使用 `FashionMNIST` 和 `ToTensor()`，将图像转为 `[1, 28, 28]` 的 float32 张量，像素缩放到 `[0, 1]`。
- 官方训练集 60000 张，使用 `random_split` 分为训练 54000 张、验证 6000 张；官方测试集 10000 张保持独立。
- 使用 `torch.Generator().manual_seed(config.seed)` 固定数据划分。这里只保证划分可重复，尚未实现整个训练过程的随机性控制。
- 校验 `0 < config.val_size < len(full_train_dataset)`，不满足时抛出 `ValueError`。
- 训练加载器 `shuffle=True`，验证和测试 `shuffle=False`，均为 `num_workers=0`。
- 两次创建 `FashionMNIST` 都使用 `download=True`。

### 模型中的已知问题

`model.py` 中有：

```python
# 现有问题代码：末尾逗号会生成单元素元组。
self.input_size = input_size,
```

已实际确认传入 `784` 后存储的是 `(784,)`，类型为 `tuple`。进入模型步骤时解释并修正。当前还没有确定最终网络结构；可以从简单全连接分类器开始教学，但应明确这是下一步建议，而非已经实现或约定的模型。

## 4. 当前停在何处：最后一个数据文件下载失败

用户最新一次运行 `python train.py`：先成功输出配置，并显示三个下载进度 100%，随后报错：

```text
RuntimeError: Error downloading t10k-labels-idx1-ubyte.gz:
Tried http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/, got:
HTTP Error 502: Bad Gateway
```

报错发生在 `dataset.py` 第一次创建 `FashionMNIST(train=True, ...)` 时。该类的下载流程会准备整套训练与测试文件，所以此时请求测试集标签是正常行为。

交接前已检查 `D:\code\26fall_AiLearning\projects\FashionMNIST_review\data\FashionMNIST\raw`，当前仅有：

| 文件 | 字节数 |
|---|---:|
| `train-images-idx3-ubyte.gz` | 26421880 |
| `train-images-idx3-ubyte` | 47040016 |
| `train-labels-idx1-ubyte.gz` | 29515 |
| `train-labels-idx1-ubyte` | 60008 |
| `t10k-images-idx3-ubyte.gz` | 4422102 |
| `t10k-images-idx3-ubyte` | 7840016 |

**仍缺 `t10k-labels-idx1-ubyte.gz` 及其解压文件。数据加载尚未验收通过。**

### 已验证的恢复方案

助手已从数据发布方仓库的以下地址读取缺失文件到内存，确认当时可访问、大小为 5148 字节，MD5 与 torchvision 中的预期值一致：

- 地址：[发布方仓库中的测试集标签](https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/t10k-labels-idx1-ubyte.gz)
- 预期 MD5：`bb300cfdad3c16e7a12a480ee83cd310`

该检查没有向项目写入文件。以下命令已在上轮聊天交给用户，**用户尚未反馈执行结果**：

```powershell
# 指向发布方仓库中的测试集标签压缩包。
$labelUrl = 'https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/t10k-labels-idx1-ubyte.gz'

# 保存到 torchvision 正在使用的数据目录。
$labelArchive = 'D:\code\26fall_AiLearning\projects\FashionMNIST_review\data\FashionMNIST\raw\t10k-labels-idx1-ubyte.gz'

# -Uri 指定下载地址；-OutFile 指定保存文件。
Invoke-WebRequest -Uri $labelUrl -OutFile $labelArchive

# 核对文件内容；显示结果的大小写不影响 MD5 比较。
Get-FileHash -LiteralPath $labelArchive -Algorithm MD5

# 确认 Hash 一致后运行；torchvision 会识别已有压缩包并解压。
python train.py
```

无需删除前三个已下载的数据文件。优先完成这一小步，不因下载端点故障重写数据模块或更换数据集。

## 5. 数据步骤的验收标准

重新运行入口，除配置外应看到：

```text
训练集样本数: 54000
验证集样本数: 6000
测试集样本数: 10000
图像形状: torch.Size([64, 1, 28, 28])
标签形状: torch.Size([64])
图像类型: torch.float32
标签类型: torch.int64
```

这是预期输出，不是已经得到的运行结果。必须以用户新反馈或实际运行结果确认后，才把数据步骤标为完成。

## 6. 后续指导顺序

1. 先查看缺失标签是否已经补齐，或接收用户执行恢复命令后的结果，完成数据验收。
2. 再进入 `model.py`：介绍 `nn.Module`、继承、`super().__init__()`、可训练层和 `forward()`；修正原有逗号问题。先用真实批次验证 `[64, 1, 28, 28]` 到 `[64, 10]` 的输出形状。
3. 讲解分类输出、标签、交叉熵、优化器、清空梯度、反向传播和参数更新，先跑通单个批次，再写完整训练循环。
4. 加入验证流程，区分 `model.train()`、`model.eval()` 和关闭梯度的作用；按样本数正确汇总 loss 与 accuracy。
5. 根据验证集指标保存最佳 checkpoint，加入日志；测试集留给独立评估，不用于选择最佳模型。
6. 编写独立 `eval.py`，再补齐实际环境依赖、运行命令、结果和已知问题等 README 内容。

以上是教学接续路线，不表示这些代码已经存在或通过测试。用户需要时再逐步展开，不要一次性给出整套成品。

## 7. 已讲过的概念与常见误区

- `@dataclass` 自动生成初始化等方法，类型标注用于说明与辅助检查，不会自动强制类型转换。
- 类外的 `describe_config(config)` 是普通函数，不应额外写 `self` 参数。
- 配置属性通过 `config.batch_size` 等方式访问，裸写 `batch_size` 会产生 `NameError`。
- `return` 将值交给调用者，`print` 输出到终端；没有显式返回值的函数默认返回 `None`。
- f 字符串、相邻字符串拼接、`\n`、`if __name__ == "__main__"`。
- Dataset 管理单个样本，DataLoader 将样本组织为批次。
- `iter()`、`next()`、元组解包、张量的 `shape` 与 `dtype`。
- `len(loader.dataset)` 是样本数，`len(loader)` 是批次数。
- `Path`、`__file__`、`resolve()`、`parent` 和 `/` 路径拼接。

这些内容已解释过，但不代表用户已经完全掌握；后续应结合实际代码简短复习。

## 8. 给后续聊天的启动指令

> 请阅读这份交接文档，并检查 `D:\code\26fall_AiLearning\projects\FashionMNIST_review` 的最新文件。从当前数据下载问题接着指导我；如果数据检查已通过，就进入模型前向传播。继续让我逐步自己写和运行，每轮给一个能验收的小步骤。所有新文件写文件头用途说明，每个代码模块有简要注释，少见的接口和语法解释其功能。不要假定模型、训练、验证或 checkpoint 已完成。

## 9. 参考来源

项目属于父级 Git 仓库 `D:\code\26fall_AiLearning`，远程为 [zhaoming9955/26fall_AiLearning](https://github.com/zhaoming9955/26fall_AiLearning)，项目位于 `projects/FashionMNIST_review`。用户已授权将此项目提交并推送，其他项目的未提交和已暂存修改不属于本次提交范围；最终推送状态以 GitHub 或 Git 检查为准。

- [原周计划](D:/code/26fall_AiLearning/weekly/week01/README.md)
- [PyTorch 官方数据教程](https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html)
- [ToTensor 文档](https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.ToTensor.html)
- [torch 与 torchvision 官方版本配对](https://pytorch.org/get-started/previous-versions/#v2110)
- [FashionMNIST 发布方仓库与数据校验值](https://github.com/zalandoresearch/fashion-mnist)
