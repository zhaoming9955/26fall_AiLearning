# FashionMNIST 项目复习清单与掌握程度测试

> 目的：把“跟着代码运行成功”转化为“能够理解、解释、修改和重新实现”。
>
> 建议先闭卷完成概念和代码阅读题，再完成排错与编程题。不要查看当前项目代码，也不要直接让 AI 生成答案。

## 一、这个项目中值得复习的可复用能力

### 1. 必须独立掌握的 Python 基础

#### 1.1 导入模块

```python
from pathlib import Path
from model import FashionClassifier
```

需要理解：

- 为什么要把项目拆分为多个文件；
- `import` 后可以使用哪些名称；
- 导入名称拼错为什么会报错。

#### 1.2 函数

应当能解释并编写函数的参数、返回值、类型标注和局部变量，例如：

```python
def choose_device(device_name: str) -> torch.device:
    """根据配置选择计算设备。"""
```

#### 1.3 类和对象

需要理解：

- `FashionClassifier` 是类；
- `model` 是实例（对象）；
- `__init__()` 定义初始化过程；
- `self.network` 是实例属性；
- `forward()` 是方法；
- `FashionClassifier()` 是实例化。

#### 1.4 主程序入口

```python
if __name__ == "__main__":
    main()
```

需要理解为什么直接运行文件时执行 `main()`，而从其他文件导入时不应该自动开始训练。

#### 1.5 字典与 dataclass

需要理解：

- YAML 被解析后为什么是字典；
- `TrainConfig` 为什么比到处使用字典更清晰；
- `config.batch_size` 的值从哪里来。

#### 1.6 文件路径与异常

```python
checkpoint_path = Path("checkpoints/best_model.pt")

if not checkpoint_path.exists():
    raise FileNotFoundError("找不到模型文件")
```

这是机器学习工程和普通 Python 工程都会反复使用的能力。

### 2. 必须掌握的 PyTorch 主线

不需要背下所有 API，但必须理解下面的完整流程：

```text
准备数据
→ 建立模型
→ 前向传播
→ 计算损失
→ 清空梯度
→ 反向传播
→ 更新参数
→ 验证模型
→ 保存最佳模型
→ 测试模型
```

#### 2.1 Tensor 和形状

对于图片批次形状：

```text
[64, 1, 28, 28]
```

应当能解释：

- `64`：图片数量；
- `1`：灰度通道数；
- `28 × 28`：每张图片的高度和宽度。

#### 2.2 Dataset 和 DataLoader

需要理解：

- Dataset 管理单个样本；
- DataLoader 负责将样本组成批次；
- 为什么训练集需要 `shuffle=True`；
- Batch 和 Epoch 的区别。

#### 2.3 模型定义

```python
class FashionClassifier(nn.Module):
    """FashionMNIST 分类模型。"""
```

需要理解：

- 为什么继承 `nn.Module`；
- 网络层为什么在 `__init__()` 中创建；
- 数据为什么在 `forward()` 中流动；
- 为什么模型最终为每张图片输出10个数。

#### 2.4 训练五步

以下五行是最值得反复默写和理解的内容：

```python
optimizer.zero_grad()  # 清除上一批次的梯度。
logits = model(images)  # 执行前向传播。
loss = loss_function(logits, labels)  # 计算损失。
loss.backward()  # 通过反向传播计算梯度。
optimizer.step()  # 根据梯度更新模型参数。
```

#### 2.5 训练集、验证集和测试集

- 训练集：用于更新模型参数；
- 验证集：用于观察效果、比较方案和选择最佳模型；
- 测试集：只用于最后一次客观评估。

#### 2.6 训练模式与评估模式

```python
model.train()
model.eval()
```

以及：

```python
@torch.no_grad()
```

需要理解这些操作的用途，而不只是记住写法。

#### 2.7 保存和加载模型

需要理解：

- `state_dict()` 保存的是什么；
- `torch.save()` 做什么；
- `torch.load()` 做什么；
- 为什么加载参数前必须创建结构相同的模型。

### 3. 必须掌握的工程习惯

这些能力可以迁移到以后的深度学习、Agent 和 VLA 项目：

- 为每个项目创建独立虚拟环境；
- 用 YAML 保存容易调整的参数；
- 按照功能拆分代码文件；
- 使用固定随机种子提高可复现性；
- 把模型检查点保存到 `checkpoints/`；
- 阅读 Traceback，优先查看最后一行错误；
- 每完成一个小模块就单独运行验证；
- 完成一个阶段后提交 Git；
- 在 README 中记录安装方法和运行方法。

### 4. 目前不要求熟练掌握的内容

现阶段暂时不需要：

- 背诵所有 PyTorch API；
- 手工推导完整的反向传播公式；
- 深入研究 CUDA Toolkit 和显卡驱动的所有兼容细节；
- 编写复杂 CNN；
- 分布式训练；
- 混合精度训练；
- 一次性默写整个项目。

现阶段目标是：**理解完整流程，并能在参考文档的情况下独立重新实现。**

---

## 二、掌握程度测试

总分100分。建议按照顺序完成，不要先查看项目代码。

## 第一级：概念测试（30分）

每题3分，请使用自己的话回答。

### 1. `FashionClassifier` 和 `model = FashionClassifier()` 分别是什么？

答：FashionClassifier是类的定义, 而model=FashionClassifier()是实例化


### 2. 一批图片的形状是 `[64, 1, 28, 28]`，四个数字各表示什么？

答：不知道


### 3. Batch 和 Epoch 有什么区别？

答：Batch是一批有多少个样本, 而Epoch是训练的周期数


### 4. 为什么训练集通常设置 `shuffle=True`，测试集却不需要？

答：训练的时候避免数据顺序影响模型的训练水平


### 5. `logits = model(images)` 实际会调用模型中的哪个方法？

答：FashionClassifier的main方法


### 6. 下面五步的正确顺序是什么？为什么？

```text
optimizer.step()
loss.backward()
optimizer.zero_grad()
loss = loss_function(logits, labels)
logits = model(images)
```

答：optimizer.zero_grad() -> model(image) -> optimizer.step() -> loss = loss_function ->


### 7. 训练集、验证集和测试集分别有什么用途？

答：让模型训练,提升模型水平. 评估模型训练水平. 评估模型训练水平


### 8. `model.train()` 与 `model.eval()` 有什么区别？

答：一个用于训练, 需要打乱样本, 使用训练集.另一个用于测试, 不需要打乱样本, 使用的是测试集


### 9. `@torch.no_grad()` 为什么用于验证和测试？

答：因为模型只有训练的时候才需要梯度积累和回传. 而验证,测试的时候只需要检验模型水平


### 10. 为什么保存 `model.state_dict()`，而不是只保存测试准确率？

答：不知道


## 第二级：代码阅读测试（20分）

阅读以下代码：

```python
for images, labels in data_loader:
    # 将数据移动到模型所在设备。
    images = images.to(device)
    labels = labels.to(device)

    # 使用模型计算类别分数。
    logits = model(images)

    # 在类别维度上选出分数最高的位置。
    predictions = logits.argmax(dim=1)

    # 统计预测正确的数量。
    correct_predictions += (
        predictions == labels
    ).sum().item()
```

每题4分。

### 1. `images.to(device)` 解决了什么问题？

答：将图片送入内存中


### 2. `logits` 是类别编号，还是10个类别的分数？

答：10个类别的分数


### 3. 为什么使用 `argmax(dim=1)`？

答：不知道


### 4. `predictions == labels` 得到的是什么？

答：True


### 5. `.sum().item()` 分别完成什么操作？

答：累加, item()不清楚


## 第三级：排错测试（20分）

下面的代码有至少5处错误。请指出错误并修改：

```python
class FashionClassifier(nn.Module):
    """一个存在错误的分类模型。"""

    def init(self):
        """创建模型层。"""
        super().init()

        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forword(self, images):
        """执行前向传播。"""
        self.network(images)


model = FashionClassifier
images = torch.randn(64, 1, 28, 28)
logits = model(images)
```

评分方法：发现一处错误得2分，正确修改该错误再得2分。

修改后的代码：

```python
# 在这里填写修改后的完整代码，并给关键操作添加简要注释。
```

## 第四级：独立编程测试（30分）

新建一个练习文件：

```text
exercises/rebuild_evaluate.py
```

不要复制项目中的 `evaluate()`。独立编写下面的函数：

```python
def evaluate(model, data_loader, loss_function, device):
    """在指定数据集上计算平均损失和准确率。"""
```

函数必须做到：

1. 切换到评估模式；
2. 关闭梯度计算；
3. 遍历所有批次；
4. 把图片和标签移动到指定设备；
5. 完成前向传播；
6. 计算每批损失；
7. 统计正确预测数量；
8. 统计总样本数；
9. 返回平均损失和准确率。

测试限制：

- 可以查阅 PyTorch 官方文档；
- 不可以打开当前项目中的 `train.py`；
- 不可以直接让 AI 生成答案；
- 建议在40分钟内完成；
- 完成后再与原来的 `evaluate()` 对照。

评分参考：

- 正确设置评估模式和关闭梯度：5分；
- 正确遍历数据并移动设备：5分；
- 正确执行前向传播和计算损失：5分；
- 正确统计样本数和预测正确数：5分；
- 正确计算并返回平均损失与准确率：5分；
- 命名清晰、缩进正确、注释合理：5分。

---

## 三、评分标准

| 分数 | 当前掌握程度 | 建议 |
|---:|---|---|
| 0～39 | 主要停留在抄写阶段 | 重新梳理 Python 函数、类和 Tensor |
| 40～59 | 能理解部分代码 | 重做训练五步和评估函数 |
| 60～74 | 掌握基本流程 | 可以继续项目，同时查漏补缺 |
| 75～89 | 能独立解释和修改 | 已达到本项目的学习目标 |
| 90～100 | 能够独立迁移 | 可以开始 CNN 或下一个项目 |

## 四、最终合格标准

真正的合格标准不是把代码全部背下来，而是：

1. 不看原代码，也能解释完整训练流程；
2. 给出代码骨架后，能够补全训练和评估逻辑；
3. 遇到简单 Traceback，能够定位报错文件、行号和原因；
4. 能把相同流程迁移到另一个简单分类数据集；
5. 知道自己不理解什么，并能查阅文档解决问题。

建议先完成第一级和第二级，再把答案交给我逐题评分。评分时会优先提供提示，让你自己修正，而不是立刻给出完整答案。
