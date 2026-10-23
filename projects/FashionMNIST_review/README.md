<!-- 文件作用：说明 FashionMNIST 分类项目的结构、环境、运行方法、已验收结果和已知限制。 -->

# FashionMNIST_review

Week 1 的 FashionMNIST 分类项目复现。使用全连接网络完成训练、验证、最佳模型保存、日志记录与独立测试。

## 当前进度

- [x] 配置创建、数据加载与训练/验证/测试划分。
- [x] 用户自行编写模型、单批次更新、多轮训练与验证。
- [x] 按验证准确率保存最佳 checkpoint，恢复后重新验证指标一致。
- [x] 日志完整验收：模型说明、每轮指标、最佳模型保存提示和结束摘要均写入文件。
- [x] 独立运行 eval.py，加载磁盘 checkpoint 评估测试集。
- [x] 更新本说明与 HANDOFF.md，记录真实运行结果。
- [x] 补齐 config.py 文件头，并移除 utils.py 文档字符串后的多余反斜杠。
- [x] 完成最终训练、最佳模型保存、完整日志和独立评估验收。

所有 Python 源码由助手在聊天中提供带注释版本，用户自行写入；助手可检查和运行代码，更新 Markdown 文档。

## 项目结构

| 文件 | 作用 |
|---|---|
| config.py | dataclass 训练配置与配置摘要 |
| dataset.py | FashionMNIST 下载、ToTensor、数据划分与 DataLoader |
| model.py | Flatten → Linear(784, 128) → ReLU → Linear(128, 10) |
| train.py | 训练、逐轮验证、最佳模型选择与日志 |
| eval.py | 可复用评估函数和独立测试入口 |
| utils.py | 同时输出到终端和文件的日志记录器 |
| requirements.txt | 核心依赖版本 |
| HANDOFF.md | 后续聊天的接续说明 |

## 环境

2026-10-23 在本电脑实际确认：

- Python 3.11.16，解释器为 D:\Anaconda\envs\cqu-ai\python.exe。
- torch 2.11.0+cu128、torchvision 0.26.0+cu128。
- 当前训练和独立评估均使用 CPU。

在另一台电脑重建环境时，可使用 requirements.txt 指定的版本。CPU 构建足以运行当前配置。先克隆父仓库，再进入 projects/FashionMNIST_review：

~~~powershell
# 克隆并进入项目目录。
git clone https://github.com/zhaoming9955/26fall_AiLearning.git
Set-Location '.\26fall_AiLearning\projects\FashionMNIST_review'

# 创建学习环境；已存在同名环境时直接激活。
conda create -n cqu-ai python=3.11 -y
conda activate cqu-ai

# 安装 CPU 构建的核心依赖。
python -m pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cpu
~~~

若需要与本机相同的 CUDA 12.8 构建，将安装命令的软件源改为 https://download.pytorch.org/whl/cu128。

## 运行

在项目目录和 cqu-ai 环境中执行：

~~~powershell
# 从头训练，逐轮验证并保存本次运行中的最佳模型。
python -X utf8 train.py

# 单独读取已经保存的最佳模型，评估测试集，不执行训练。
python -X utf8 eval.py

# 查看训练日志的末尾。
Get-Content -LiteralPath '.\logs\train.log' -Encoding UTF8 -Tail 20
~~~

运行产物：

- checkpoints/best.pt：本次训练运行中，验证准确率最高时的模型、优化器状态、配置、轮次和验证指标；每次刷新最佳成绩时覆盖。
- logs/train.log：追加保存的训练日志，多次运行通过“开始新的训练”标记区分。
- data/FashionMNIST/raw/：下载及解压的数据。

checkpoint 和数据均不通过 Git 同步；在其他电脑上可以重新下载数据、重新训练。再次运行训练会开始一个新模型，并覆盖此前的 best.pt。

## 数据与模型

- 训练集 54000 张，验证集 6000 张，测试集 10000 张。
- 使用 seed=42 固定训练/验证划分。
- ToTensor 将图像转换为形状 [1, 28, 28] 的 float32 张量，并将像素缩放到 [0, 1]。
- 模型输出 10 个类别的原始分数，直接交给 CrossEntropyLoss。
- batch_size=64，Adam 学习率 0.001，epochs=5。
- 按样本数汇总平均损失，准确率为正确预测数除以总样本数。

## 已验收结果

以下结果来自 2026-10-23 同一次最终验收的 checkpoint：训练于 08:48:52 开始，08:49:37 结束，随后独立运行测试。

| 指标 | 数值 |
|---|---:|
| 保存轮次 | 5 |
| 本轮训练损失 | 0.3249 |
| 本轮训练准确率 | 88.31% |
| 保存的验证损失 | 0.3459 |
| 保存的验证准确率 | 87.52% |
| 测试损失 | 0.3665 |
| 测试准确率 | 87.09% |

测试结果来自独立运行 eval.py，测试样本数为 10000。本表记录本次验收结果；以后重新训练时，需要将本表更新为新 checkpoint 对应的结果，避免混用不同运行的数据。

## 已知限制与下载恢复

- seed 目前只用于固定数据划分，尚未统一固定模型初始化与训练批次顺序，所以重新训练的结果可能不同。
- 已保存优化器状态，但尚未编写断点续训入口。
- main() 的目标说明被放在 train.py 末尾入口调用之后，原函数说明仍是旧描述；属于可整理的注释位置问题，不影响已验收的运行行为。
- 如果从 base 环境运行出现 OpenMP 重复加载错误，应切换到 cqu-ai 或明确使用项目解释器。
- 默认数据下载服务器曾返回 HTTP 502。可从发布方仓库补齐当时缺失的测试集标签，保留其余已下载文件：

~~~powershell
# 确保数据目录存在。
New-Item -ItemType Directory -Force -Path '.\data\FashionMNIST\raw' | Out-Null

# 从发布方仓库下载缺失的标签压缩包。
$labelUrl = 'https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/t10k-labels-idx1-ubyte.gz'
$labelArchive = '.\data\FashionMNIST\raw\t10k-labels-idx1-ubyte.gz'
Invoke-WebRequest -Uri $labelUrl -OutFile $labelArchive

# 核对预期 MD5：BB300CFDAD3C16E7A12A480EE83CD310。
Get-FileHash -LiteralPath $labelArchive -Algorithm MD5
~~~

确认校验值一致后再运行训练，torchvision 会处理解压与读取。

## 注释约定

新增 Python 文件先写文件头，说明文件作用；每个功能模块有简要注释，较少见的接口和语法就近解释功能。

