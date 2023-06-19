
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from init import *
from pre import *

dump = np.load('dump.npy', allow_pickle=True).item()

xs = dump['x']
ys = dump['y']
ws = dump['w']

layer = 8 # 0 / 4 / 8
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

print( np.allclose(o, y) )
print ( equal / total )

###########################################

x = np.reshape(x, (b * h * w, k * k * c, 1))
mul = x * f

print (np.shape(x))
print (np.shape(f))
print (np.shape(mul))

###########################################

mul = np.transpose(mul, (0, 2, 1))
mul = np.reshape(mul, (b * h * w * n, k * k * c))
print (np.shape(mul))

###########################################

scales = []
for m in mul:
    vals = np.abs(m)
    vals = vals[np.where(m > 0)]
    scale = np.max(vals) / vals
    scale = np.log2(scale)
    scale = np.mean(scale)
    scales.append(scale)

print (np.mean(scales))

###########################################



