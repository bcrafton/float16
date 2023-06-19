
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from init import *

##############################

cifar10 = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

y_train = y_train.astype(np.int32)

x_train = np.reshape(x_train, (-1, 50, 32, 32, 3))
y_train = np.reshape(y_train, (-1, 50))

x_train = x_train - np.mean(x_train)
x_train = x_train / np.max(x_train)

##############################

class conv:
    def __init__(self, k, c, n):
        f = init_filters([k, k, c, n], init="glorot_uniform")
        b = np.zeros(shape=n)
        self.f = tf.Variable(f, dtype=tf.float32)
        self.b = tf.Variable(b, dtype=tf.float32)
    def train(self, x):
        y = tf.nn.conv2d(x, self.f, [1,1,1,1], 'SAME') + self.b
        return tf.nn.relu(y)
    def params(self):
        return [self.f, self.b]

##############################

class dense:
    def __init__(self, row, col, relu=False):
        self.row = row
        self.col = col
        self.relu = relu
        w = init_matrix(size=[row, col], init='glorot_uniform')
        b = np.zeros(shape=col)
        self.w = tf.Variable(w, dtype=tf.float32)
        self.b = tf.Variable(b, dtype=tf.float32)
    def train(self, x):
        x = tf.reshape(x, (-1, self.row))
        y = tf.matmul(x, self.w) + self.b
        if self.relu: out = tf.nn.relu(y)
        else:         out = y
        return out
    def params(self):
        return [self.w, self.b]

##############################

class pool:
    def __init__(self, k):
        self.k = k
    def train(self, x):
        pool = tf.nn.avg_pool(x, ksize=self.k, strides=self.k, padding="SAME")
        return pool
    def params(self):
        return []

##############################

class model:
    def __init__(self, layers):
        self.layers = layers
    def train(self, x):
        for layer in self.layers:
            x = layer.train(x)
        return x
    def params(self):
        ret = []
        for layer in self.layers:
            ret.extend( layer.params() )
        return ret

##############################

m = model([
conv(3, 3, 32),
pool(2),

conv(3, 32, 32),
pool(2),

conv(3, 32, 64),
pool(2),

dense(4*4*64, 10, relu=False)
])

##############################

params = m.params()
optimizer = tf.keras.optimizers.Adam()

##############################

def gradients(model, x, y):
    with tf.GradientTape() as tape:
        logits = model.train(x)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
        label = tf.argmax(logits, axis=1)
        correct = tf.reduce_sum(tf.cast(tf.equal(label, y), tf.float32))
    grad = tape.gradient(loss, params)
    return loss, grad, correct

##############################

for epoch in range(10):
    total_loss = 0
    total_correct = 0
    for (x, y) in zip(x_train, y_train):
        loss, grad, correct = gradients(m, x, y)
        optimizer.apply_gradients(zip(grad, params))
        total_loss += np.sum(loss)
        total_correct += correct
    print ('Epoch: %d Loss: %f Accuracy: %f' % (epoch, total_loss / 50000, total_correct / 50000))

##############################

