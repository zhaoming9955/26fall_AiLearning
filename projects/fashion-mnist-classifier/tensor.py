""" """

import torch

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"当前设备: {device}")

scalar = torch.tensor(7)

vector = torch.tensor([1, 2, 3])

matrix = torch.tensor(
    [
        [1, 2, 3],
        [4, 5, 6]
    ],
    dtype = torch.float32,
)

print("\n标量: ")
print(scalar)
print(f"维度数量：{scalar.ndim}")
print(f"形状：{scalar.shape}")


print("\n向量：")
print(vector)
print(f"维度数量：{vector.ndim}")
print(f"形状：{vector.shape}")


print("\n矩阵：")
print(matrix)
print(f"维度数量：{matrix.ndim}")
print(f"形状：{matrix.shape}")
print(f"数据类型：{matrix.dtype}")
print(f"所在设备：{matrix.device}")


matrix = matrix.to(device)

print("\n移动设备后：")
print(matrix)
print(f"所在设备：{matrix.device}")