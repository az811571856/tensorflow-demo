# coding:utf-8
# 多项式回归
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
# 定义初始参数
w_target = np.array([0.5, 3, 2.4])
b_target = np.array([0.9])
# 定义原始数据和函数
x_sample = np.arange(-3, 3.1, 0.5)  # 原始数据很多x点
y_sample = b_target[0] + w_target[0] * x_sample + w_target[1] * x_sample ** 2 + w_target[2] * x_sample ** 3  # 原始数据很多y点
# 画出函数的曲线
# plt.plot(x_sample, y_sample, label='real curve')
# plt.legend()
# plt.show() # 默认不显示
# x_sample 构造x**1 x**2 x**3 数据， 2维， shape: (3, x_sample.length) -> (x_sample.length, 3)
# [x_sample ** i for i in range(1, 4)] x_sample先进行 ** 运算，生成x_sample ** 数组，再把生成的数组循环三次，生成三个数组，三个数组被最外层数组包裹
x_train = np.stack([x_sample ** i for i in range(1, 4)], axis=1)
# 构造原始数据 tensor
x_train = tf.constant(x_train, dtype=tf.float32, name='x_train')
y_train = tf.constant(y_sample, dtype=tf.float32, name='y_train')
# 构造参数变量
w = tf.Variable(initial_value=tf.random_normal(shape=(3, 1), dtype=tf.float32), name='weights')  # 随机初始值
b = tf.Variable(initial_value=0.0, name='bias')
# 构造模型函数 y_预测值
y_ = tf.squeeze(tf.matmul(x_train, w) + b)  # 预测数据很多y_点
# 计算图
sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())
# 画出原始和初始预测曲线 默认不显示
# plt.plot(x_train_value[:, 0], y_pred_value, label='fitting value', color='r')
# plt.plot(x_train_value[:, 0], y_train_value, label='real value', color='b')
# plt.legend()
# plt.show()

# 定义损失函数，梯度函数，求梯度
loss = tf.reduce_mean(tf.square(y_ - y_train))
w_grad, b_grad = tf.gradients(loss, [w, b])
# 更新参数
lr = 5e-3 #速率太大会晃动永远找不到收敛
w_update = w.assign_sub(lr * w_grad)
b_update = b.assign_sub(lr * b_grad)

# 运行梯度下降
for i in range(100):
    sess.run([w_update, b_update])
    x_train_value = sess.run(x_train)
    y_train_value = sess.run(y_train)
    y_pred_value = sess.run(y_)  # 预测值
    # 画图
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.ion()
    fig.show()
    fig.canvas.draw()
    ax.clear()
    ax.plot(x_train_value[:, 0], y_pred_value, label='fitting value', color='r')
    ax.plot(x_train_value[:, 0], y_train_value, label='real value', color='b')
    ax.legend()
    fig.canvas.draw()
    plt.pause(0.5)
    fig.clear()


