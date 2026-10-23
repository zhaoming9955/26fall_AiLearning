<!-- 文件作用：向后续聊天交接本项目的教学约定、代码状态、真实验收结果和剩余收尾任务。 -->

# FashionMNIST_review 聊天交接

更新日期：2026-10-23（Asia/Shanghai）。本文件替代早先分散的历史进度说明；接续时仍需检查用户是否保存了新改动。

## 用户目标与协作方式

- 只推进 Week 1 的 FashionMNIST 分类项目复现。
- 周计划：D:\code\26fall_AiLearning\weekly\week01。
- 当前项目：D:\code\26fall_AiLearning\projects\FashionMNIST_review。
- 用户明确要求：Python 源码只由助手在聊天中提供带注释版本，用户自己写入；禁止助手直接创建或修改项目 Python 文件。
- 可以检查代码、执行终端命令、运行验收、更新 Markdown。用户近期希望自己运行主要命令，指导时提供可复制的 PowerShell 指令。
- 新文件有文件头用途说明，功能模块有简要注释，陌生接口与语法就近解释。
- 每完成一个阶段，报告已完成与未完成项。
- 不要求固定论文、额外网络结构或目标准确率，不为凑代码行数扩展功能。

## 环境与 Git

本电脑实际确认的环境：

- Python：D:\Anaconda\envs\cqu-ai\python.exe，版本 3.11.16。
- torch 2.11.0+cu128，torchvision 0.26.0+cu128。
- 当前训练配置和独立评估都使用 CPU。
- 从 base 运行曾出现 OpenMP 重复加载错误，项目环境运行正常；不要把 base 解释器误当成项目环境。
- 另一台电脑曾使用不同路径和 CPU 构建；不要将其路径套用到本电脑。

父级仓库：D:\code\26fall_AiLearning。

远程：https://github.com/zhaoming9955/26fall_AiLearning.git，分支 main。

此前本项目已通过 49b0cca 推送，随后从另一台电脑同步过 3241492。最终收尾版本的提交号以 git log 为准。父仓库存在其他项目的已暂存、未暂存和未跟踪内容，提交时必须只包含 projects/FashionMNIST_review，并保留其他改动及其暂存状态。禁止使用全仓库 git add 或不限定范围的提交。

项目 .gitignore 排除数据、日志、checkpoint、输出目录和缓存，因此另一台电脑 clone 后需要自行生成数据及模型文件。

## 当前实现

| 模块 | 当前行为 |
|---|---|
| config.py | TrainConfig：batch_size=64、learning_rate=0.001、epochs=5、device=cpu、data_dir=data、val_size=6000、seed=42 |
| dataset.py | ToTensor，官方训练集拆为 54000/6000，测试集 10000；固定划分种子，训练加载器打乱，num_workers=0 |
| model.py | nn.Module，Flatten(start_dim=1)，784→128→10，ReLU，返回 logits |
| train.py | train_one_epoch，逐轮验证，以验证准确率选择 best.pt，全部运行信息已接入 logger |
| eval.py | evaluate 使用 model.eval 与 torch.no_grad；main 加载 best.pt，在 CPU 上独立测试 |
| utils.py | setup_logger 同时输出终端和 logs/train.log，追加模式，UTF-8 |
| requirements.txt | torch==2.11.0、torchvision==0.26.0 |
| README.md | 已于本次更新环境、结构、训练/评估命令、当前指标、限制和剩余清单 |

checkpoint 字段为 epoch、model_state_dict、optimizer_state_dict、val_loss、val_acc、config。

训练每次创建新模型；best_val_acc 每次初始化 -1，best.pt 保存的是本次运行的最佳模型，不是所有历史运行中的最佳模型。

## 验收证据

- 数据加载、前向传播、单批次更新、5 轮训练与验证均已运行。
- 曾恢复 checkpoint 并重新计算验证指标，与保存值一致。
- 当前 best.pt 来自用户 2026-10-23 08:48:52 开始、08:49:37 结束的最终训练，第 5 轮最佳。
- 该 checkpoint 的训练损失 0.3249、训练准确率 88.31%、验证损失 0.3459、验证准确率 87.52%。
- 用户已独立运行 eval.py：加载第 5 轮模型，测试集 10000 张，测试损失 0.3665、测试准确率 87.09%。
- 本次所有项目 .py 文件通过 ast 语法解析。
- logs/train.log 的最后一次运行包含全部 5 轮指标、模型说明、最佳保存提示和结束摘要；已根据本地日志核对。
- config.py 文件头已补齐，utils.py 文档字符串后的多余反斜杠已删除。
- train.py 中 main() 的新说明被用户贴在末尾入口调用之后，旧函数说明还在；这是不影响运行的注释位置问题，后续可由用户自行整理。

不要把更早的 87.83% 或 86.80% 验证准确率当成当前 checkpoint 结果，旧 best.pt 已被后续训练覆盖。

## 接续建议

本次要求的数据、模型、训练、验证、最佳 checkpoint、日志、独立测试及结果文档已完成。后续先读取本文件和 Git 状态，不要重复要求用户实现已完成模块，也不要未经用户要求扩展网络、调参或重跑训练。

可选整理：由用户将 train.py 末尾的说明文字移入原 main() 的 docstring，再删除末尾那条孤立字符串；无需新增函数或改变执行流程。

重新训练会覆盖 best.pt。若用户后续运行新实验，使用同一次训练与独立测试的结果更新 README；当前验收指标无需再重复训练。

## 后续命令

在项目目录执行：

~~~powershell
# 激活环境后，从头训练，再独立测试。
conda activate D:\Anaconda\envs\cqu-ai
python -X utf8 train.py
python -X utf8 eval.py

# 检查最终日志。
Get-Content -LiteralPath '.\logs\train.log' -Encoding UTF8 -Tail 30
~~~

助手若不依赖终端激活，可明确使用 D:\Anaconda\envs\cqu-ai\python.exe。

## 边界

目前 seed 只固定划分，模型初始化和训练数据顺序未统一固定；新运行指标变化是可能的。已保存优化器状态但未实现恢复训练入口。CLI 参数、曲线图、混淆矩阵、额外网络、调参都不是本次必须完成的要求。

