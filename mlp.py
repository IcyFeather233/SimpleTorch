import matplotlib.pyplot as plt

from simple_torch import *


# 定义一个分类器类，继承自Module类
class Classifier(Module):
    # 初始化函数，接收输入维度din，中间维度dmid和输出维度dout
    def __init__(self, din: int, dmid: int, dout: int):
        # 初始化第一个线性层，将输入维度din映射到中间维度dmid
        self.linear1 = Linear(din, dmid)
        # 初始化第二个线性层，将中间维度dmid映射到输出维度dout
        self.linear2 = Linear(dmid, dout)

    # 调用函数，用于执行分类器的前向传播
    def __call__(self, x: Array) -> Array:
        # 将输入x传递给第一个线性层
        x = self.linear1(x)
        # 对第一个线性层的输出应用ReLU激活函数
        x = relu(x)
        # 将ReLU后的输出传递给第二个线性层
        x = self.linear2(x)
        # 返回第二个线性层的输出结果
        return x


# 设置每个类别的数据点数量
N = 100
# 设置数据的维度，即每个数据点的特征数
D = 2
# 设置类别的数量
K = 3

# 初始化输入数据数组，形状为(N*K, D)，初始值都为0
inputs = np.zeros((N * K, D))
# 初始化标签数组，长度为N*K，数据类型为无符号8位整数
labels = np.zeros(N * K, dtype="uint8")

# 循环生成每个类别的数据点
for j in range(K):
    # 计算当前类别数据点的索引范围
    ix = range(N * j, N * (j + 1))

    # 为当前类别生成半径值，均匀分布在[0.0, 1.0]区间
    r = np.linspace(0.0, 1.0, N)

    # 角频率倍数
    alpha = 4
    # 为当前类别生成theta值，基础值是每个类别4倍的角频率，加上小的高斯噪声
    t = np.linspace(j * alpha, (j + 1) * alpha, N) + np.random.randn(N) * 0.2

    # 根据半径和theta值生成二维坐标点，存储到inputs数组中
    # np.c_是numpy的列堆叠操作，用于生成(x, y)坐标点
    inputs[ix] = np.c_[r * np.sin(t), r * np.cos(t)]

    # 将当前类别的标签赋值给labels数组
    labels[ix] = j

# 使用matplotlib库对生成的数据进行可视化
# plt.scatter用于生成散点图，其中：
# inputs[:, 0]是x轴坐标，inputs[:, 1]是y轴坐标
# c=labels指定了每个点的颜色，根据标签的值变化
# s=40设置点的大小
plt.scatter(inputs[:, 0], inputs[:, 1], c=labels, s=40)

# plt.show()用于显示生成的图形
plt.title(f"Generated Data: {D} dimensions, {K} categories, {N} points for each")
plt.show()


# 使用Array类将输入数据和标签封装起来，以适应自动微分框架
inputs = Array(inputs)
labels = Array(labels)

# 创建一个分类器模型实例，其中：
# 2是输入数据的特征数量（dimensionality）
# 64是中间层的大小（dmid）
# K是输出类别的数量（dout）
model = Classifier(2, 64, K)

# 获取模型的所有参数，并转换为列表
params = list(model.parameters())

print("params of Classifier: ", params)

# 设置学习率（步长）为0.001
step_size = 1e-3

# 进行10,000次迭代的训练过程
for i in range(10_000 + 1):
    # 通过模型对输入数据进行前向传播，得到logits
    logits = model(inputs)

    # 计算交叉熵损失，并使用mean函数将其转换为标量值
    loss = mean(cross_entropy_with_logits(logits, labels))

    # 执行反向传播，计算损失相对于模型参数的梯度
    loss.backward()

    # 每100次迭代打印一次训练信息
    if i % 100 == 0:
        # 使用模型对输入数据进行预测，获取预测得分
        scores = model(inputs).value

        # 根据预测得分，找到预测类别
        predicted_class = np.argmax(scores, axis=1)

        # 计算模型预测的准确率
        accuracy = np.mean(predicted_class == labels.value)

        # 打印当前迭代次数、损失值和准确率
        print(f"iteration {i}: {loss.value = :.4f}, {accuracy = :.4f}")

    # 更新模型的参数：使用梯度下降法
    for p in model.parameters():
        # 参数的值减去学习率乘以梯度，实现参数更新
        p.value += -step_size * p.grad

    # 重置loss的梯度，为下一次迭代做准备
    loss.zero_grad()


# 绘制决策边界
# 为x轴创建一个线性空间，范围是输入数据中第一个特征的最小值和最大值，共100个点
x = np.linspace(inputs.value[:, 0].min(), inputs.value[:, 0].max(), 100)
# 为y轴创建一个线性空间，范围是输入数据中第二个特征的最小值和最大值，共100个点
y = np.linspace(inputs.value[:, 1].min(), inputs.value[:, 1].max(), 100)
# 使用np.meshgrid生成一个网格，覆盖x和y的范围
xx, yy = np.meshgrid(x, y)
# 将网格的点堆叠成一个新的数组，每个点表示为一个二维坐标（x, y）
grid_inputs = np.stack([xx, yy], axis=-1)

# 使用分类器模型对网格中的每个点进行预测
grid_preds = model(Array(grid_inputs)).value

# 对每个点的预测结果找到最大值的索引，即预测的类别
grid_labels = np.argmax(grid_preds, axis=-1)

# 使用matplotlib的contourf函数绘制决策边界的填充图
plt.contourf(xx, yy, grid_labels)

# 使用scatter函数绘制原始数据点的散点图
# c=labels.value指定点的颜色，根据标签的值变化
# s=40设置点的大小
# edgecolors="k"设置点的边缘颜色为黑色
plt.scatter(
    inputs.value[:, 0], inputs.value[:, 1], c=labels.value, s=40, edgecolors="k"
)

# 显示绘制的图形
plt.show()
