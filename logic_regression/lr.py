# coding: utf-8
# 逻辑回归
# 读取数据
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
with open('logistic_regression.txt', 'r') as f:
    data_list = [i.split("\n")[0].split(',') for i in f.readlines()]
    data = [(float(j[0]), float(j[1]), float(j[2])) for j in data_list]

# 标准化
x0_max = max(k[0] for k in data)
x1_max = max(k[1] for k in data)
data = [(i[0]/x0_max, i[1]/x1_max, i[2]) for i in data] # [100, 3]

# 画图展示第一类第二类点
x0 = list(filter(lambda x: x[-1] == 0, data))  # label 为0的点
plot_x0 = [i[0] for i in x0]
plot_y0 = [i[1] for i in x0]
x1 = list(filter(lambda x: x[-1] == 1, data))  # label 为1的点
plot_x1 = [i[0] for i in x1]
plot_y1 = [i[1] for i in x1]
plt.plot(plot_x0, plot_y0, 'ro', label='x_0')
plt.plot(plot_x1, plot_y1, 'bo', label='x_1')
# plt.show()

# 构建模型
np_data = np.array(data, dtype='float32')
x_data = tf.constant(np_data[:, 0:2])  # [100, 2] 包含第一行第二行的原始数据
y_data = tf.constant(np.expand_dims(np_data[:, -1], axis=1))  # [100, 1] expand_dims: [100,] -> [100,1] 包含label数据
w = tf.Variable(tf.random_normal([2, 1], seed=2017))  # [2, 1]
b = tf.Variable(tf.zeros([1]))
y_pred = tf.sigmoid(tf.matmul(x_data, w) + b)  # [100, 1]

# 画出模型的分割线
sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())

w_numpy = sess.run(w)
b_numpy = sess.run(b)
w0 = w_numpy[0]
w1 = w_numpy[1]
b0 = b_numpy[0]
plot_x = np.arange(0.2, 1, 0.01)
plot_y = (-w0 * plot_x - b0) / w1
plt.plot(plot_x, plot_y, 'g', label='cutting line')
# plt.show()
# 损失函数
loss = -tf.reduce_mean(y_data*tf.log(y_pred) + (1-y_data)*tf.log(1-y_pred))  # 真实值与预测值相同时，损失函数最小

# 自定义的优化器
# w_grad, b_grad = tf.gradients(loss, [w, b])
# lr = 5
# w_update = w.assign_sub(w_grad * lr)
# b_update = b.assign_sub(b_grad * lr)

# tensorflow的优化器
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1, name='myOptimizer')
train_op = optimizer.minimize(loss)

for i in range(10200):
    # w_numpy, b_numpy = sess.run([w_update, b_update]) # 自定义的优化器

    print("i: {}, loss : {}".format(i, sess.run(loss)))
    # if i % 20 == 0:
    #     w0 = w_numpy[0]
    #     w1 = w_numpy[1]
    #     b0 = b_numpy[0]
    #     plot_x = np.arange(0.2, 1, 0.01)
    #     plot_y = (-w0 * plot_x - b0) / w1
    #     plt.plot(plot_x, plot_y, 'g', label='cutting line')
    #     plt.legend()
        # plt.show()

