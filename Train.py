import tensorflow as tf
import numpy as np
from math import sqrt
import Operations as op

inp = np.float32(np.random.normal(0, 1, (1, 1, 100, 100)))
conv1_weights = np.float32(np.random.normal(0, 0.1, (5, 1, 3, 3)))
conv2_weights = np.float32(np.random.normal(0, 0.1, (10, 5, 3, 3)))
gamma1, beta1 = np.float32(np.random.normal(0, 0.1, 2))
gamma2, beta2 = np.float32(np.random.normal(0, 0.1, 2))



'''   LAYER 1   '''
conv1 = op.convolve(inp, conv1_weights, pad=1)
relu1 = op.relu(conv1)
batch_norm1 = op.batch_normalize(relu1, gamma1, beta1)

'''   LAYER 2   '''
conv2 = op.convolve(batch_norm1, conv2_weights, pad=1)
relu2 = op.relu(conv2)
batch_norm2 = op.batch_normalize(relu2, gamma2, beta2)
pool2 = op.pool(batch_norm2, 2, 2, 2)


sess = tf.InteractiveSession()
w1 = tf.constant(conv1_weights.transpose(2, 3, 1, 0))
w2 = tf.constant(conv2_weights.transpose(2, 3, 1, 0))
i = inp.transpose(0, 2, 3, 1)

w1 = tf.cast(w1, dtype=tf.float32)
w2 = tf.cast(w2, dtype=tf.float32)
i = tf.cast(i, dtype=tf.float32)

c1 = tf.nn.conv2d(i, w1, strides=[1, 1, 1, 1], padding='SAME')
r1 = tf.nn.relu(c1)
mean, var = tf.nn.moments(r1, axes=[0, 1, 2])
b1 = tf.nn.batch_normalization(r1, mean, var, beta1, gamma1, 1e-8)

c2 = tf.nn.conv2d(b1, w2, strides=[1, 1, 1, 1], padding='SAME')
r2 = tf.nn.relu(c2)
mean, var = tf.nn.moments(r2, axes=[0, 1, 2])
b2 = tf.nn.batch_normalization(r2, mean, var, beta2, gamma2, 1e-8)
p = tf.nn.max_pool(b2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

out = sess.run(p).transpose(0, 3, 1, 2)

print(out.shape)
print(pool2.shape)
print("****************out***************")
print(out)
print("**************pool2*****************")
print(pool2)
print(np.array_equal(pool2, out))