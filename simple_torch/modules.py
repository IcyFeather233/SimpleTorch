from typing import Any, Callable
import numpy as np
from collections import deque
import scipy
import scipy.special

from .array import Array


# 定义一个基础模块类，用于构建神经网络的组件
class Module:
    # 参数生成器方法，用于迭代当前模块及其子模块的所有参数
    def parameters(self):
        # vars(self)返回一个包含类实例所有属性的字典
        for value in vars(self).values():
            # 如果属性是Array类型，直接yield
            if isinstance(value, Array):
                yield value
            # 如果属性是Module类型，递归调用其parameters方法
            elif isinstance(value, Module):
                # yield from递归地yield子模块的参数
                yield from value.parameters()


# 定义一个线性层类，继承自Module类
class Linear(Module):
    # 初始化函数，接受输入维度din和输出维度dout
    def __init__(self, din: int, dout: int):
        # 初始化权重w为一个Array对象，其值从[0, 0.01)区间内均匀分布的随机数
        self.w = Array(np.random.uniform(0, 0.01, size=(din, dout)))
        # 初始化偏置b为一个Array对象，其值为0
        self.b = Array(np.zeros((dout,)))

    # 调用函数，用于执行线性层的前向传播
    def __call__(self, x: Array) -> Array:
        # 执行矩阵乘法并加上偏置b，返回结果
        return x @ self.w + self.b
