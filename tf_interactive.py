# -*- coding: utf-8 -*

import tensorflow as tf

sess = tf.InteractiveSession()
a = tf.constant(1)
b = tf.constant(2)
c = a + b
# 这里不用 sess.run(c)
c.eval()
print(c.eval())