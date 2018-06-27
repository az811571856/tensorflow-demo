# coding:utf8
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
tf.set_random_seed(2017)


# 绘制决策边界
def plot_decision_boundary(model, x, y):
    # 找到x, y的最大值和最小值, 并在周围填充一个像素
    x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    h = 0.01
    # 构建一个宽度为`h`的网格
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # 计算模型在网格上所有点的输出值
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # 画图显示
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(x[:, 0], x[:, 1], c=np.squeeze(y), cmap=plt.cm.Spectral)
    # 不显示
    plt.show()


np.random.seed(100)
# 400个样本，平均分成两类
m = 400 # 样本数量
N = int(m/2) # 每一类的点的个数
D = 2 # 维度
# [400,2]
x = np.zeros((m, D))
# [400,1]
y = np.zeros((m, 1), dtype='uint8') # label 向量，0 表示红色，1 表示蓝色
a = 4

#   j: 0,1
for j in range(2):
    # 200 个: 0-199 or 200-399
    ix = range(N*j,N*(j+1))
    # shape 200
    t = np.linspace(j*3.12,(j+1)*3.12,N) + np.random.randn(N)*0.2 # theta
    # shape 200
    r = a*np.sin(4*t) + np.random.randn(N)*0.2 # radius
    # =号右侧：shpae 200,2
    x[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
    # shape
    y[ix] = j

plt.scatter(x[:, 0], x[:, 1], c=np.squeeze(y), s=40, cmap=plt.cm.Spectral)
# 不显示
# plt.show()
x = tf.constant(x, dtype=tf.float32, name='x')
y = tf.constant(y, dtype=tf.float32, name='y')

# 定义模型
w = tf.get_variable(initializer=tf.random_normal_initializer(), shape=(2, 1), dtype=tf.float32, name='weights')
b = tf.Variable(tf.constant(0.0))


def logistic_model(x):
    logit = tf.matmul(x, w) + b

    return tf.sigmoid(logit)


y_ = logistic_model(x)

# 构造训练
loss = tf.losses.log_loss(predictions=y_, labels=y)

lr = 1e-1
optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
train_op = optimizer.minimize(loss)

sess = tf.InteractiveSession()

sess.run(tf.global_variables_initializer())

for e in range(10000):
    sess.run(train_op)
    if (e + 1) % 1000 == 0:
        loss_numpy = loss.eval(session=sess)
        print('Epoch %d: Loss: %.12f' % (e + 1, loss_numpy))

model_input = tf.placeholder(shape=(None, 2), dtype=tf.float32, name='logistic_input')
logistic_output = logistic_model(model_input)


def plot_logistic(x_data):
    y_pred_numpy = sess.run(logistic_output, feed_dict={model_input: x_data})
    out = np.greater(y_pred_numpy, 0.5).astype(np.float32)
    return np.squeeze(out)


plot_decision_boundary(plot_logistic, x.eval(session=sess), y.eval(session=sess))


# 神经网络代码
# 第一层神经网路
with tf.variable_scope('layer1'):
    # 构建参数weight
    w1 = tf.get_variable(initializer=tf.random_normal_initializer(stddev=0.01), shape=(2, 4), name='weights1')

    # 构建参数bias
    b1 = tf.get_variable(initializer=tf.zeros_initializer(), shape=(4), name='bias1')

# 同样地, 我们再构建第二个隐藏层
with tf.variable_scope('layer2'):
    w2 = tf.get_variable(initializer=tf.random_normal_initializer(stddev=0.01), shape=(4, 1), name='weights2')
    b2 = tf.get_variable(initializer=tf.zeros_initializer(), shape=(1), name='bias2')

# 通过上面的参数构建一个两层的神经网络
def two_network(nn_input):
    with tf.variable_scope('two_network'):
        # 第一个隐藏层
        net = tf.matmul(nn_input, w1) + b1
        # tanh 激活层
        net = tf.tanh(net)
        # 第二个隐藏层
        net = tf.matmul(net, w2) + b2

        # 经过 sigmoid 得到输出
        return tf.sigmoid(net)


net = two_network(x)

# 构建神经网络的训练过程
loss_two = tf.losses.log_loss(predictions=net, labels=y, scope='loss_two')

lr = 1

optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
train_op = optimizer.minimize(loss=loss_two, var_list=[w1, w2, b1, b2])

# 模型的保存和训练
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())

for e in range(10000):
    sess.run(train_op)
    if (e + 1) % 1000 == 0:
        loss_numpy = loss_two.eval(session=sess)
        print('Epoch {}: Loss: {}'.format(e + 1, loss_numpy))
    if (e + 1) % 5000 == 0:
        # `sess`参数表示开启模型的`session`, 必选参数
        # `save_path`参数表示模型保存的路径, 必须要以`.ckpt`结尾
        # `global_step`参数表示模型当前训练的步数, 可以用来标记不同阶段的模型
        saver.save(sess=sess, save_path='First_Save/model.ckpt', global_step=(e + 1))



nn_out = two_network(model_input)

def plot_network(input_data):
    y_pred_numpy = sess.run(nn_out, feed_dict={model_input: input_data})
    out = np.greater(y_pred_numpy, 0.5).astype(np.float32)
    return np.squeeze(out)

plot_decision_boundary(plot_network, x.eval(session=sess), y.eval(session=sess))
plt.title('2 layer network')
