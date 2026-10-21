<!-- 文件作用：说明 FashionMNIST_review 的学习进度、跨电脑克隆与环境配置方法。 -->

# FashionMNIST_review

Week 1 的 FashionMNIST 分类项目复现，按步骤学习配置、数据加载、模型、训练、验证、checkpoint 和日志。

## 当前进度

- 配置创建与输出已运行成功。
- 已编写训练、验证和测试数据加载，以及批次形状检查。
- 数据加载上次运行因测试集标签下载地址返回 HTTP 502 而中断；需按下文补齐文件后重新验证。
- `model.py` 仍为普通 Python 类练习；训练循环、验证指标、模型保存和独立评估尚未完成。
- `python train.py` 当前执行配置与数据检查。

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
