# Simple Torch

Simple Torch 是一个迷你版 PyTorch，包含以下内容：

- 一个自定义的 Array 类及使用 numpy 实现的基本运算操作，已经与 JAX 进行了对比测试
- 一个 Module 类及线性层（Linear layer）
- 一个两层多层感知器（MLP）的示例


目录结构：

```
.
├── README.md
├── mlp.py                  # mlp 示例
├── requirements.txt
├── simple_torch            # simple_torch 包
│   ├── __init__.py
│   ├── __pycache__
│   ├── array.py
│   └── modules.py
└── test.py                 # 对 simple_torch 包基本功能的测试

```

## Installation

```
conda create -n simple-torch python=3.9
conda activate simple-torch
pip install -r requirements.txt
```

## Run

测试 simple_torch 的基本功能：

```
python test.py
```

测试 MLP 示例代码：

```
python mlp.py
```


