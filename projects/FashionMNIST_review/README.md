<!-- 文件作用：说明 FashionMNIST_review 的学习进度、跨电脑克隆与环境配置方法。 -->

# FashionMNIST_review

Week 1 的 FashionMNIST 分类项目复现，按步骤学习配置、数据加载、模型、训练、验证、checkpoint 和日志。

## 当前进度

- 配置创建与输出已运行成功。
- 2026-10-21 已在本机 `cqu-ai` 环境中通过依赖导入、张量运算及 `pip check` 检查。
- 数据下载与加载已运行通过：训练集 54000 张、验证集 6000 张、测试集 10000 张；图像批次形状为 `[64, 1, 28, 28]`。
- `model.py` 已由用户自行抄写完成：继承 `nn.Module`，使用展平和 `784 -> 128 -> 10` 全连接网络，输出类别原始分数；真实批次 `[64, 1, 28, 28] -> [64, 10]` 及单张图像输出已运行验收。
- `train.py` 当前执行配置与数据检查。
- 助手此前代写的 Python 修改已按用户要求撤回；所有 Python 代码由助手在聊天中提供带注释版本，用户自己抄写，终端操作由助手执行。

## 任务清单

- [x] 本机环境配置、依赖检查。
- [x] 配置模块运行验收。
- [ ] 补齐 `config.py` 文件头说明（用户自行抄写）。
- [x] 数据下载、划分、批次检查。
- [x] 可训练模型定义与真实批次前向传播验收（用户自行抄写完成）。
- [ ] 在 `train.py` 接入模型并验收入口输出。
- [ ] 单批次训练：损失、优化器、梯度清零、反向传播、参数更新。
- [ ] 完整训练循环与验证指标。
- [ ] 最佳模型保存与日志。
- [ ] 独立评估与最终结果说明。

## 当前电脑的运行环境

- 项目路径：`D:\CODE\26Fall\26fall_AiLearning\projects\FashionMNIST_review`。
- 解释器：`D:\Anaconda_Folder\envs\cqu-ai\python.exe`。
- Python 3.11.17、torch 2.11.0+cpu、torchvision 0.26.0+cpu、Pillow 12.3.0。
- 当前 PowerShell 的默认 `python` 仍指向 base；助手执行项目命令时明确使用 `cqu-ai`，后续终端操作由助手完成。

```powershell
# conda run -n 指定使用哪个环境，无需依赖当前终端是否已激活该环境。
# --no-capture-output 让程序输出直接显示在终端。
conda run --no-capture-output -n cqu-ai python train.py
```

下一步：用户根据聊天中的带注释代码在 `train.py` 接入模型，再由助手运行入口验收；随后学习单批次训练。每个文件或一批文件完成后，对照任务清单报告进度。当前模型尚未训练，前向传播通过不代表已获得分类准确率。

## 在另一台电脑继续

在 PowerShell 中选择自己的代码保存目录，再执行：

```powershell
# 克隆父仓库；本项目位于其 projects 子目录。
git clone https://github.com/zhaoming9955/26fall_AiLearning.git
Set-Location '.\26fall_AiLearning\projects\FashionMNIST_review'

# 创建并激活学习环境；如果同名环境已存在，直接激活即可。
conda create -n cqu-ai python=3.11 -y
conda activate cqu-ai

# 安装核心依赖。CPU 构建足以运行当前配置为 cpu 的学习项目。
python -m pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cpu

# 输出配置、样本数量及一个批次的形状；首次运行需要下载数据。
python train.py
```

原电脑使用 Python 3.11.16、torch 2.11.0+cu128、torchvision 0.26.0+cu128。若需要同样的 CUDA 12.8 构建，在新环境中将安装命令的软件源改为 `https://download.pytorch.org/whl/cu128`。

## 下载问题恢复

如果默认下载地址返回 502，可从发布方仓库补齐测试集标签。先在项目目录执行：

```powershell
# 确保保存目录存在；-Force 允许目录已经存在。
New-Item -ItemType Directory -Force -Path '.\data\FashionMNIST\raw' | Out-Null

# 指定发布方文件地址与本地保存位置。
$labelUrl = 'https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/t10k-labels-idx1-ubyte.gz'
$labelArchive = '.\data\FashionMNIST\raw\t10k-labels-idx1-ubyte.gz'

# 下载压缩包，并检查 MD5 是否为 BB300CFDAD3C16E7A12A480EE83CD310。
Invoke-WebRequest -Uri $labelUrl -OutFile $labelArchive
Get-FileHash -LiteralPath $labelArchive -Algorithm MD5

# 确认校验值一致后，交给 torchvision 解压并继续加载。
python train.py
```

数据检查的预期结果为：训练集 54000 张、验证集 6000 张、测试集 10000 张；默认批次图像形状 `[64, 1, 28, 28]`，标签形状 `[64]`。

## 跨电脑同步

下载数据、Python 缓存、日志和模型产物通过 `.gitignore` 排除，需要时在各电脑重新生成。

已有克隆的电脑可以在工作区没有未提交改动时运行 `git pull --ff-only` 获取最新提交。完整学习交接见 [HANDOFF.md](HANDOFF.md)。

## 代码注释约定

新增文件先写文件头，解释文件作用；每个代码模块给出简要注释，较少见的接口和语法注明功能。
