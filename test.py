# 测试部分，使用JAX库验证自定义操作的正确性
import jax.numpy as jnp
import jax
import optax

from simple_torch import *


# 定义测试函数，验证自定义mean函数的反向传播是否正确
def test_mean():
    # 创建一个Array实例a，包含5个从0到4的浮点数
    a = Array(np.arange(5, dtype=np.float32))

    # 计算a的均值，并将结果作为损失值
    loss = mean(a)

    # 执行反向传播，计算a的梯度
    loss.backward()

    # JAX部分，用于验证自定义梯度
    # 定义一个函数f，计算输入数组x的均值
    def f(x):
        return jnp.mean(x)

    # 使用jax.grad计算函数f的梯度
    grad = jax.grad(f)(jnp.arange(5, dtype=jnp.float32))

    # 使用numpy的assert_allclose断言来比较自定义梯度和JAX计算的梯度是否足够接近
    np.testing.assert_allclose(a.grad, grad)
    print("Mean Test Passed!")


# 定义测试函数，验证自定义sum函数的反向传播是否正确
def test_sum():
    # 创建一个Array实例a，包含5个从0到4的浮点数
    a = Array(np.arange(5, dtype=np.float32))

    # 计算a的总和，并将结果作为损失值
    loss = sum(a)

    # 执行反向传播，计算a的梯度
    loss.backward()

    # JAX部分，用于验证自定义梯度
    # 定义一个函数f，计算输入数组x的总和
    def f(x):
        return jnp.sum(x)

    # 使用jax.grad计算函数f的梯度
    grad = jax.grad(f)(jnp.arange(5, dtype=jnp.float32))

    # 使用numpy的assert_allclose断言来比较自定义梯度和JAX计算的梯度是否一致
    np.testing.assert_allclose(a.grad, grad)
    print("Sum Test Passed!")


# 定义测试函数，验证自定义broadcast函数的反向传播是否正确
def test_broadcast():
    # 创建一个Array实例a，包含5个从0到4的浮点数
    a = Array(np.arange(5, dtype=np.float32))

    # 对a进行广播操作，将其形状从(5,)变为(2, 5)
    b = broadcast(a, (2, 5))

    # 计算广播后的数组b的总和，并将结果作为损失值
    loss = sum(b)

    # 执行反向传播，计算a的梯度
    loss.backward()

    # JAX部分，用于验证自定义梯度
    # 定义一个函数f，计算输入数组x的广播后的总和
    def f(x):
        # jnp.broadcast_to实现数组的广播，x[None]将x变为(1, 5)形状
        return jnp.sum(jnp.broadcast_to(x[None], (2, 5)))

    # 使用jax.grad计算函数f的梯度
    grad = jax.grad(f)(jnp.arange(5, dtype=jnp.float32))

    # 使用numpy的assert_allclose断言来比较自定义梯度和JAX计算的梯度是否足够接近
    np.testing.assert_allclose(a.grad, grad)
    print("Broadcast Test Passed!")


# 定义测试函数，验证自定义矩阵乘法（点乘）操作的反向传播是否正确
def test_dot():
    # 创建一个Array实例a，其值为5x3x4的随机数数组
    a = Array(np.random.uniform(size=(5, 3, 4)))
    # 创建一个Array实例b，其值为4x2的随机数数组
    b = Array(np.random.uniform(size=(4, 2)))

    # 执行矩阵乘法操作，得到结果c
    # 这里a的形状是(5, 3, 4)，b的形状是(4, 2)，根据矩阵乘法规则，结果c的形状将是(5, 3, 2)
    c = a @ b

    # 计算c的总和，并将结果作为损失值
    loss = sum(c)

    # 执行反向传播，计算a和b的梯度
    loss.backward()

    # JAX部分，用于验证自定义梯度
    # 定义一个函数f，接受两个参数t，这里t是一个元组，包含a和b
    def f(t):
        # 从t中解包出a和b
        a, b = t
        # 执行矩阵乘法操作，得到c
        c = a @ b
        # 返回c的总和
        return jnp.sum(c)

    # 使用jax.grad计算函数f关于a和b的梯度
    # jnp.array将自定义Array实例的.value属性转换为JAX的jnp.array类型
    grad_a, grad_b = jax.grad(f)((jnp.array(a.value), jnp.array(b.value)))

    # 使用numpy的assert_allclose断言来比较自定义梯度和JAX计算的梯度是否足够接近
    # rtol=1e-5设置相对容差为0.00001
    np.testing.assert_allclose(a.grad, grad_a, rtol=1e-5)
    np.testing.assert_allclose(b.grad, grad_b, rtol=1e-5)
    print("Dot Test Passed!")


# 定义测试函数，验证自定义线性层操作的反向传播是否正确
def test_linear():
    # 创建一个Array实例x，其值为5x3x4的随机数数组
    x = Array(np.random.uniform(size=(5, 3, 4)), name="x")
    # 创建一个Array实例w，其值为4x2的随机数数组，表示权重
    w = Array(np.random.uniform(size=(4, 2)), name="w")
    # 创建一个Array实例b，其值为长度为2的随机数数组，表示偏置
    b = Array(np.random.uniform(size=(2,)), name="b")

    # 执行线性变换操作，即矩阵乘法加上偏置，得到结果y
    y = x @ w + b

    # 计算y的总和，并将结果作为损失值
    loss = sum(y)

    # 执行反向传播，计算x, w, 和 b的梯度
    loss.backward()

    # JAX部分，用于验证自定义梯度
    # 定义一个函数f，接受权重和偏置作为参数，以及输入x
    def f(params, x):
        # 从params元组中解包出权重w和偏置b
        w, b = params
        # 执行线性变换操作，即矩阵乘法加上偏置，得到y
        y = x @ w + b
        # 返回y的总和
        return jnp.sum(y)

    # 使用jax.grad计算函数f关于w和b的梯度
    # jnp.array将自定义Array实例的.value属性转换为JAX的jnp.array类型
    grad_w, grad_b = jax.grad(f)(
        (jnp.array(w.value), jnp.array(b.value)), jnp.array(x.value)
    )

    # 使用numpy的assert_allclose断言来比较自定义梯度和JAX计算的梯度是否足够接近
    # rtol=1e-5设置相对容差为0.00001
    np.testing.assert_allclose(w.grad, grad_w, rtol=1e-5)
    np.testing.assert_allclose(b.grad, grad_b, rtol=1e-5)
    print("Linear Test Passed!")


# 定义测试函数，验证自定义交叉熵损失函数的反向传播是否正确
def test_cross_entropy_with_logits():
    # 创建一个Array实例logits，其值为5x10的随机数数组，表示未经过softmax转换的原始预测值
    logits = Array(np.random.uniform(size=(5, 10)))
    # 创建一个Array实例labels，其值为长度为5的数组，包含从0到9的整数，表示正确的类别标签
    labels = Array(np.random.randint(0, 10, size=(5,)))

    # 计算交叉熵损失，并将结果作为损失值
    loss = mean(cross_entropy_with_logits(logits, labels))

    # 执行反向传播，计算logits的梯度
    loss.backward()

    # JAX部分，用于验证自定义损失函数和梯度
    # 定义一个函数f，接受logits和labels作为参数，计算交叉熵损失的均值
    def f(logits, labels):
        # 使用optax库的softmax_cross_entropy_with_integer_labels函数计算交叉熵损失
        return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, labels))

    # 使用jax.value_and_grad函数同时获取损失值和梯度
    # 对于logits和labels，将自定义Array实例的.value属性转换为JAX的jnp.array类型
    loss_jax, grad = jax.value_and_grad(f)(logits.value, labels.value)

    # 使用numpy的assert_allclose断言来比较自定义损失值和JAX计算的损失值是否足够接近
    # 使用相同的断言来比较自定义梯度和JAX计算的梯度是否足够接近
    # rtol=1e-5设置相对容差为0.00001
    np.testing.assert_allclose(loss.value, loss_jax, rtol=1e-5)
    np.testing.assert_allclose(logits.grad, grad, rtol=1e-5)
    print("Cross-Entropy Test Passed!")


def run_tests():
    test_mean()
    test_sum()
    test_broadcast()
    test_dot()
    test_linear()
    test_cross_entropy_with_logits()
    print("All Tests Passed!")


run_tests()
