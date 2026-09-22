# 14 周路线

每周核心任务 26 小时，另留算法练习 5 小时、复盘 2 小时、缓冲 2 小时。日期按实际开学进度顺延。

| 周 | 核心任务（26 小时） | 最低验收 |
| --- | --- | --- |
| 01 | Python 类、文件与异常 10h；JSON 实验记录本 10h；终端、虚拟环境和 Git 6h | 新增、查询、保存记录；用 status/diff/add/commit/push 完成一次上传 |
| 02 | 测试与模块 6h；搜索和 CSV 导出 12h；Git 历史、忽略规则、分支 4h；VLA 简介 4h | 处理空输入和文件不存在；完成一个小分支并合并 |
| 03 | NumPy 8h；Tensor 和自动求导 8h；线性回归 8h；机器人数据概念 2h | 画损失曲线，解释张量形状与梯度 |
| 04 | PyTorch 教程 10h；Fashion-MNIST 训练和加载 12h；数据划分 4h | 独立训练、保存、加载与评估 |
| 05 | 注意力概念 8h；小张量实验 8h；SmolVLA 阅读 6h；版本和算力确认 4h | 一页模型流程和复现范围 |
| 06 | LeRobot 环境 6h；轨迹读取和可视化 12h；行为克隆和划分 6h；整理 2h | 图像与动作对应，按完整轨迹划分数据 |
| 07 | 推理示例 6h；真实样本推理 14h；预处理和输出理解 6h | 重复运行成功，输出形状正确且无 NaN |
| 08 | 训练配置 6h；小规模微调 14h；保存与恢复 3h；求职准备 3h | 训练可运行，checkpoint 可加载，记录实际资源消耗 |
| 09 | 固定验证集对比 12h；失败分析 6h；报告 5h；求职 3h | 完整实验报告，明确离线指标的局限 |
| 10 | Agent 基础 6h；一次可靠 API 调用 10h；VLA 5h；求职 5h | 密钥从环境读取，调用异常可处理 |
| 11 | RAG 概念 5h；关键词检索与来源 12h；VLA 5h；求职 4h | 20 个评测问题，包含资料无答案情况 |
| 12 | 工具调用 5h；两个工具与执行循环 12h；VLA 5h；求职 4h | 参数校验、调用记录、最多 3 次工具调用 |
| 13 | FastAPI 4h；接口与运行说明 10h；VLA 5h；求职 7h | 干净环境可运行，3 分钟演示 |
| 14 | 模拟面试 8h；修复与验证 6h；导师汇报 5h；求职 7h | 能解释核心代码、实验结果和真实失败经历 |

## 工具学习范围

第 1 周：终端的当前目录与相对路径；Python 解释器、虚拟环境和依赖；Git 工作区、暂存区、提交与远程仓库。先学日常操作，不学复杂历史修改。

第 2 周：`.gitignore`、`git log`、差异查看；创建一个小分支并合并。尚不学习 rebase、reset --hard 或 force push。

第 3—4 周：认识终端报错、调试器、日志、训练配置。按项目需要学习，不单独开一门大而全工具课。

## 算力安排

- 笔记本：日常开发、Python、NumPy、PyTorch 入门和 Agent 应用。
- 台式机：确认实际显存和软件环境后，测试 VLA 小批次推理及训练。
- 不把两台机器的显存直接相加；前期不安排分布式训练。
- 先跑少量数据和几十步，测量显存、速度和稳定性，再决定是否租卡。
- 租卡前确认小时单价、存储费与停止实例后的计费规则；设置预算提醒，保留应急余量。
- 使用租赁资源前准备好数据、环境说明和配置，结束后取回必要结果并停止计费资源。

## 资源（按需选读）

- Python 教材配套：https://ehmatthes.github.io/pcc_3e/
- NumPy 入门：https://numpy.org/doc/stable/user/absolute_beginners.html
- PyTorch 基础：https://docs.pytorch.org/tutorials/beginner/basics/intro.html
- 动手学深度学习：https://zh.d2l.ai/
- LeRobot：https://github.com/huggingface/lerobot
- SmolVLA：https://huggingface.co/docs/lerobot/smolvla
- Agent 课程：https://huggingface.co/learn/agents-course/unit0/introduction
- FastAPI：https://fastapi.tiangolo.com/tutorial/
- Hello 算法：https://www.hello-algo.com/

## 降级规则

连续两周完成率不足 70%，下周停止增加功能，优先补基础和导师任务。无可用算力时记录阻碍，不把未执行的实验算作完成。没有闭环评估时，不把动作误差称为任务成功率。
