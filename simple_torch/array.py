from typing import Any, Callable
import numpy as np
from collections import deque
import scipy
import scipy.special


# 定义一个Array类，用于表示张量和自动微分
class Array:
    def __init__(
        self, value: np.ndarray, *, children: tuple["Array", ...] = (), name: str = ""
    ):
        self.value = value
        self.children = children  # 存储依赖于当前Array的子Array
        self.grad = np.zeros_like(self.value, dtype=np.float32)  # 存储梯度
        self.backward_fn: Callable[[], None] = lambda: None  # 存储反向传播函数
        self.name = name  # 可选的名称，用于调试

    @property
    def shape(self):
        return self.value.shape

    def backward(self):
        # 确保Array是标量，因为只有标量可以有梯度
        assert self.shape == ()
        self.grad = np.ones_like(self.value, dtype=np.float32)
        # 使用深度优先搜索实现拓扑排序
        sorted_arrays: deque[Array] = deque()
        visited: set[Array] = set()

        def dfs(array: Array):
            visited.add(array)

            for child in array.children:
                if child not in visited:
                    dfs(child)

            # 注意这里是 append left，是逆序
            sorted_arrays.appendleft(array)

        # 运行深度优先搜索
        dfs(self)

        # 按拓扑排序顺序执行反向传播函数
        for array in sorted_arrays:
            array.backward_fn()

    # 重置梯度为0
    def zero_grad(self):
        visited: set[Array] = set()

        def dfs(array: Array):
            array.grad = np.zeros_like(array.grad)
            visited.add(array)

            for child in self.children:
                if child not in visited:
                    dfs(child)

        # 运行深度优先搜索
        dfs(self)

    # 重载运算符，使Array支持加法、乘法和矩阵乘法
    def __add__(self, other: "Array") -> "Array":
        return add(self, other)

    def __mul__(self, other: "Array") -> "Array":
        return mul(self, other)

    def __matmul__(self, other: "Array") -> "Array":
        return dot(self, other)

    def __repr__(self) -> str:
        return repr(self.value)


# 定义广播函数，用于将Array实例的形状扩展到指定的新形状
def broadcast(a: Array, shape: tuple[int, ...]):
    # 确保新形状的长度至少与原形状一样大
    assert len(shape) >= len(a.shape)

    # 计算需要添加的维度数，即新形状和原形状长度的差
    added_dims = len(shape) - len(a.shape)

    # 构造一个新向量，其中缺少的维度用1填充
    a_shape = (1,) * added_dims + a.shape

    # 初始化一个列表，用于存储广播操作改变的轴
    reduce_axis: list[int] = []

    # 遍历新形状和原形状的每个维度
    for i, (a_dim, b_dim) in enumerate(zip(a_shape, shape)):
        # 如果两个维度大小不一致
        if a_dim != b_dim:
            # 如果原维度大小不是1，则抛出错误，只有1可以广播到其他大小
            if a_dim != 1:
                raise ValueError
            else:
                # 如果原维度大小是1，则将这个轴添加到reduce_axis列表中
                reduce_axis.append(i)

    # 创建一个新的Array实例，其值是原Array值广播到新形状后的结果
    out = Array(np.broadcast_to(a.value, shape), children=(a,), name="broadcast")

    # 定义广播操作的反向传播函数
    def broadcast_backward():
        # 对out.grad进行求和，求和的轴是之前记录在reduce_axis中的轴
        grad = np.sum(out.grad, axis=tuple(reduce_axis), keepdims=True)

        # 如果添加了维度，则需要将这些维度从grad中移除
        if added_dims:
            grad = np.squeeze(grad, axis=tuple(range(added_dims)))

        # 将grad累加到原Array的梯度上
        a.grad += grad

    # 将广播操作的反向传播函数赋值给out的backward_fn属性
    out.backward_fn = broadcast_backward

    # 返回新创建的Array实例
    return out


# 定义一个函数，用于检查两个Array实例是否需要广播以匹配它们的形状
def maybe_broadcast(a: Array, b: Array) -> tuple[Array, Array]:
    # 如果两个Array的形状已经相同，不需要广播，直接返回它们
    if a.shape == b.shape:
        return a, b

    # 初始化标志位，用于判断是否需要对a或b进行广播
    broadcast_a = False
    broadcast_b = False

    # 计算两个Array形状长度的绝对差值
    shape_diff = abs(len(a.shape) - len(b.shape))

    # 根据形状长度的差值，决定需要扩展的形状
    if len(a.shape) > len(b.shape):
        # 如果a的形状更长，将b的形状扩展到a的形状长度
        b_shape = (1,) * shape_diff + b.shape
        a_shape = a.shape
    elif len(b.shape) > len(a.shape):
        # 如果b的形状更长，将a的形状扩展到b的形状长度
        a_shape = (1,) * shape_diff + a.shape
        b_shape = b.shape
    else:
        # 如果形状长度相同，但形状不同，保持原样
        a_shape = a.shape
        b_shape = b.shape

    # 遍历两个形状的维度
    for a_dim, b_dim in zip(a_shape, b_shape):
        # 如果两个维度不匹配
        if a_dim != b_dim:
            # 如果都不为1，则抛出错误，因为无法广播
            if a_dim != 1 and b_dim != 1:
                raise ValueError
            # 如果a_dim为1，表示需要对a进行广播
            elif a_dim == 1:
                broadcast_a = True
            # 如果b_dim为1，表示需要对b进行广播
            elif b_dim == 1:
                broadcast_b = True

    # 如果a需要广播，调用broadcast函数进行广播
    if broadcast_a:
        a = broadcast(a, b_shape)

    # 如果b需要广播，调用broadcast函数进行广播
    if broadcast_b:
        b = broadcast(b, a_shape)

    # 返回可能已经广播的两个Array实例
    return a, b


# 定义加法函数，用于计算两个Array实例的和
def add(a: Array, b: Array):
    # 使用maybe_broadcast函数确保a和b的形状是匹配的
    a, b = maybe_broadcast(a, b)

    # 创建一个新的Array实例，其值为a和b的元素和，记录其子节点和操作名称
    out = Array(a.value + b.value, children=(a, b), name="add")

    # 定义加法操作的反向传播函数
    def add_backward():
        # 对于加法，梯度直接复制到a和b的梯度上
        a.grad += out.grad
        b.grad += out.grad

    # 将加法操作的反向传播函数赋值给out的backward_fn属性
    out.backward_fn = add_backward

    # 返回结果Array实例
    return out


# 定义乘法函数，用于计算两个Array实例的逐元素乘积
def mul(a: Array, b: Array):
    # 使用maybe_broadcast函数确保a和b的形状是匹配的
    a, b = maybe_broadcast(a, b)

    # 创建一个新的Array实例，其值为a和b的逐元素乘积，记录其子节点和操作名称
    out = Array(a.value * b.value, children=(a, b), name="mul")

    # 定义乘法操作的反向传播函数
    def mul_backward():
        # 对于乘法，梯度需要按比例分配给a和b
        # a的梯度是b的值乘以out的梯度
        a.grad += b.value * out.grad
        # b的梯度是a的值乘以out的梯度
        b.grad += a.value * out.grad

    # 将乘法操作的反向传播函数赋值给out的backward_fn属性
    out.backward_fn = mul_backward

    # 返回结果Array实例
    return out


# 定义矩阵乘法操作的函数
def dot(a: Array, b: Array) -> Array:
    # 断言b的形状是二维的，因为矩阵乘法需要两个矩阵
    assert len(b.shape) == 2

    # 执行矩阵乘法操作，创建一个新的Array实例out
    # 这里a的形状应为(M, N)，b的形状为(N, K)，结果out的形状为(M, K)
    out = Array(a.value @ b.value, children=(a, b), name="dot")

    # 定义矩阵乘法操作的反向传播函数
    def dot_backward():
        # 如果a的维度比b多，需要先对out.grad进行平均操作
        extra_dims = len(a.shape) - len(b.shape)
        grad = out.grad
        if extra_dims:
            # 对额外的维度进行平均，以匹配b的维度
            # 使用np.mean而不是np.sum可能与期望的梯度缩放有关
            grad = np.mean(grad, axis=tuple(range(extra_dims)))

        # 计算a的梯度
        # np.swapaxes用于交换矩阵的最后两个轴，grad.T是grad的转置
        # b.value @ grad.T计算矩阵乘法，a.grad += ...将结果累加到a的梯度上
        a.grad += np.swapaxes(b.value @ grad.T, -1, -2)

        # 计算b的梯度
        # 同样使用np.swapaxes交换a.value的轴，以匹配grad的形状
        # a.value @ grad计算矩阵乘法，得到b的梯度
        b_grad = np.swapaxes(a.value, -1, -2) @ grad
        if extra_dims:
            # 如果存在额外的维度，对b_grad进行求和，以减少维度
            b_grad = np.sum(b_grad, axis=tuple(range(extra_dims)))
        # 将计算得到的b的梯度累加到b的原始梯度上
        b.grad += b_grad

    # 将矩阵乘法操作的反向传播函数赋值给out的backward_fn属性
    out.backward_fn = dot_backward

    # 返回结果Array实例
    return out


# 定义计算Array平均值的函数
def mean(a: Array) -> Array:
    # 计算a.value的平均值，并创建一个新的Array实例out来存储结果
    # 这里np.mean(a.value)将计算a.value所有元素的均值，结果是一个标量
    out = Array(np.mean(a.value), children=(a,), name="mean")

    # 定义平均值操作的反向传播函数
    def mean_backward():
        # 在反向传播时，将out的梯度均分到a的每个元素上
        # 通过将out的梯度除以a的总元素数（即a.shape的乘积）实现
        a.grad += out.grad / np.prod(a.shape)

    # 将平均值操作的反向传播函数赋值给out的backward_fn属性
    out.backward_fn = mean_backward

    # 返回结果Array实例
    return out


# 定义计算Array总和的函数
def sum(a: Array) -> Array:
    # 计算a.value的总和，并创建一个新的Array实例out来存储结果
    # 这里np.sum(a.value)将计算a.value所有元素的和，结果是一个标量
    out = Array(np.sum(a.value), children=(a,), name="sum")

    # 定义求和操作的反向传播函数
    def sum_backward():
        # 在反向传播时，将out的梯度复制到a的每个元素上
        # 因为总和操作对所有元素的梯度影响是一致的
        a.grad += out.grad

    # 将求和操作的反向传播函数赋值给out的backward_fn属性
    out.backward_fn = sum_backward

    # 返回结果Array实例
    return out


# 定义交叉熵损失函数，用于分类问题
def cross_entropy_with_logits(logits: Array, labels: Array) -> Array:
    # 计算softmax概率
    probs = scipy.special.softmax(logits.value, axis=-1)

    # 计算每个类别的对数损失
    # 在取对数之前，给 probs 数组中的每个元素添加一个小的正数（如 1e-10），以避免对数运算中的 0 值。
    epsilon = 1e-10
    loss = -np.log(probs + epsilon)

    # 根据one-hot编码的labels选择相应的损失值
    # labels.value[..., None]是为了增加一个维度以匹配probs的形状
    # [...]是高级索引，用于选择与labels对应的概率的对数
    loss = np.take_along_axis(loss, labels.value[..., None], axis=-1)[..., 0]

    # 创建一个新的Array实例out来存储损失值
    out = Array(loss, children=(logits,), name="cross_entropy_with_logits")

    # 定义交叉熵损失的反向传播函数
    def cross_entropy_backwards():
        # 根据labels选择正确类别的概率
        probs_true = np.take_along_axis(probs, labels.value[..., None], axis=-1)

        # 初始化梯度为softmax概率的副本
        grad = probs.copy()

        # 将正确类别的概率置为1（因为交叉熵的梯度是概率减去1），其他为0
        np.put_along_axis(grad, labels.value[..., None], probs_true - 1, axis=-1)

        # 将梯度按labels的总数进行归一化
        grad /= np.prod(labels.shape)

        # 将计算得到的梯度累加到logits的梯度上
        logits.grad += grad

    # 将交叉熵损失的反向传播函数赋值给out的backward_fn属性
    out.backward_fn = cross_entropy_backwards

    # 返回结果Array实例
    return out


# 定义ReLU激活函数
def relu(a: Array) -> Array:
    # 应用ReLU函数，将所有负值置为0
    out = Array(np.maximum(0, a.value), children=(a,), name="relu")

    # 定义ReLU函数的反向传播函数
    def relu_backward():
        # 只有当a的值大于0时，梯度才传递
        # 这利用了ReLU函数的导数特性：对于正数，导数为1；对于负数，导数为0
        a.grad += (a.value > 0) * out.grad

    # 将ReLU函数的反向传播函数赋值给out的backward_fn属性
    out.backward_fn = relu_backward

    # 返回结果Array实例
    return out
