import os
import struct
import numpy as np


def prepro(path, type='train'):
    p_label = os.path.join(path, '%s-labels-idx1-ubyte' % type)
    p_image = os.path.join(path, '%s-images-idx3-ubyte' % type)
    with open(p_label, 'rb') as lp:
        magic, n = struct.unpack('>II', lp.read(8))
        labels = np.fromfile(lp, dtype=np.uint8)

    with open (p_image, 'rb') as ip:
        magic, num, rows, cols = struct.unpack('>IIII', ip.read(16))
        images = np.fromfile(ip, dtype=np.uint8).reshape(len(labels), 784)

    X, y = one_hot(images, labels)

    return X, y


def one_hot(images, labels):
    im = (1 * (1 < images))
    la = np.zeros((len(labels), 10))
    for i in range(len(labels)):
        la[i][labels[i]] = 1
    return im, la

