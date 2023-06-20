
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from init import *
from pre import *

dump = np.load('dump.npy', allow_pickle=True).item()

xs = dump['x']
ys = dump['y']
ws = dump['w']

for i, w in enumerate(ws):
    if w is not None:
        print (i, np.shape(w))

layer = 7
x = xs[layer]
y = ys[layer]
f = ws[layer]

def im2col(x, k=3, s=1):
    N, H, W, C = np.shape(x)
    H_PAD = pad(H, k, 'same', 1)
    W_PAD = pad(W, k, 'same', 1)
    x_pad = np.pad(x, [(0, 0), (H_PAD, H_PAD), (W_PAD, W_PAD), (0, 0)])
    ys = []
    for h in range(H):
        y = []
        for w in range(W):
            y.append( x_pad[:, h*s : (h+k)*s, w*s : (w+k)*s, :] )
        ys.append(y)
    ys = np.transpose(ys, (2, 0, 1, 3, 4, 5))
    return ys

x = im2col(x)
b, h, w, k, k, c = np.shape(x)
x = np.reshape(x, (b, h, w, k * k * c))

k, k, c, n = np.shape(f)
f = np.reshape(f, (k * k * c, n))

o = x @ f

equal = np.sum(np.isclose(o, y))
total = np.prod(np.shape(o))

# print( np.allclose(o, y) )
# print ( equal / total )

###########################################

x = np.reshape(x, (b * h * w, k * k * c, 1))
mul = x * f

# print (np.shape(x))
# print (np.shape(f))
# print (np.shape(mul))

###########################################

mul = np.transpose(mul, (0, 2, 1))
mul = np.reshape(mul, (b * h * w * n, k * k * c))
# print (np.shape(mul))

###########################################

max_x = np.max(x, axis=(1, 2))
max_f = np.max(f, axis=0)

avg_max_x = np.mean(max_x)
avg_max_f = np.mean(max_f)

# print (avg_max_x, avg_max_f)

###########################################

scales = []

np.random.shuffle(mul)
for i, m in enumerate(mul[0:5000]):
    vals = m.astype(np.float64)

    total = np.abs(np.sum(vals))
    if total > 0:
        total = np.log2(total)
    else:
        total = 0

    '''
    if i < 4:
        plt.hist(vals, bins=50)
        plt.savefig('%d.png' % (i))
        plt.cla()
        plt.clf()
    '''

    vals = np.abs(vals)

    vals = np.where(vals > 0, np.log2(vals), 0)
    vals = np.where(vals == 0, 0, vals)

    scale = total - vals

    '''
    if i < 4:
        plt.hist(scale)
        plt.savefig('%d.png' % (i))
        plt.cla()
        plt.clf()
    '''

    if i < 10:
        plt.hist(scale[np.where(scale > 10)], color='red', bins=range(20))
        plt.hist(scale[np.where(scale <= 10)], color='black', bins=range(20))
        plt.xticks(range(0, 20, 2))
        plt.savefig('%d.png' % (i))
        plt.cla()
        plt.clf()

    scales.extend(scale)

scales = np.array(scales)
print (np.mean(scales))
print (np.mean( scales > 10 ))

###########################################



