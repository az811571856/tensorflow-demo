# coding:utf-8
# 神经网络识别手写数字
import input_data
import matplotlib.pyplot as plt
import tensorflow as tf

# 读取数据，分为训练集，测试集
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
train_set = mnist.train
test_set = mnist.test


# 创建网格子图： 6*2 个，用来放置手写图片
fig, axes = plt.subplots(ncols=6, nrows=2)
# 调整子图间隔
plt.tight_layout(w_pad=-2.0, h_pad=-8.0)

# 取一些进行图形化 images:12,784  labels:12,10
images, labels = train_set.next_batch(12)

# 逐个提取
# zip:[([784],[10]), ().... ()]   12个tuple
for ind, (image, label) in enumerate(zip(images, labels)):
    # image 1，784 -> 28,28
    image = image.reshape((28, 28))

    # 取出代表的数字
    label = label.argmax()
    row = ind // 6
    col = ind % 6
    axes[row][col].imshow(image, cmap='gray')
    axes[row][col].axis('on')
    axes[row][col].set_title('%d' % label)


def hidden_layer(layer_input, output_depth, scope='hidden_layer', reuse=None):
    """
    构造单个隐藏层

    :param layer_input: 输入层
    :param output_depth:  隐藏层个数
    :param scope:
    :param reuse:
    :return:
    """
    input_depth = layer_input.get_shape()[-1]  # 输入层个数
    with tf.variable_scope(scope, reuse=reuse):
        w = tf.get_variable(initializer=tf.truncated_normal_initializer(stddev=0.1),
                            shape=(input_depth, output_depth),
                            name='weights')
        b = tf.get_variable(initializer=tf.constant_initializer(0.1),
                            shape=(output_depth),
                            name='bias')
        net = tf.matmul(layer_input, w) + b
        return net
