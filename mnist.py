import os, struct
from array import array as pyarray
from numpy import arange, append, array, int8, uint8, zeros, float64

def load_mnist(dataset="training", digits=arange(10), path="."):
    """
    Loads MNIST files into 3D numpy arrays

    Adapted from: http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
    via: http://g.sweyla.com/blog/2012/mnist-numpy/
    """

    if dataset == "training":
        fname_img = os.path.join(path, 'train-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
    elif dataset == "testing":
        fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    flbl = open(fname_lbl, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = pyarray("b", flbl.read())
    flbl.close()

    fimg = open(fname_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = pyarray("B", fimg.read())
    fimg.close()

    ind = [ k for k in range(size) if lbl[k] in digits ]
    N = len(ind)

    images = zeros((N, rows * cols), dtype=float64)
    labels = zeros((N, 1), dtype=int8)
    for i in range(len(ind)):
        images[i] = array(img[ ind[i]*rows*cols : (ind[i]+1)*rows*cols ])
        labels[i] = lbl[ind[i]]

    return images, labels

mnist_substrate = [(-13,-13), (-13,-12), (-13,-11), (-13,-10), (-13,-9), (-13,-8), (-13,-7),
(-13,-6), (-13,-5), (-13,-4), (-13,-3), (-13,-2), (-13,-1), (-13,0), (-13,1),
(-13,2), (-13,3), (-13,4), (-13,5), (-13,6), (-13,7), (-13,8), (-13,9),
(-13,10), (-13,11), (-13,12), (-13,13), (-13,14), (-12,-13), (-12,-12),
(-12,-11), (-12,-10), (-12,-9), (-12,-8), (-12,-7), (-12,-6), (-12,-5),
(-12,-4), (-12,-3), (-12,-2), (-12,-1), (-12,0), (-12,1), (-12,2), (-12,3),
(-12,4), (-12,5), (-12,6), (-12,7), (-12,8), (-12,9), (-12,10), (-12,11),
(-12,12), (-12,13), (-12,14), (-11,-13), (-11,-12), (-11,-11), (-11,-10),
(-11,-9), (-11,-8), (-11,-7), (-11,-6), (-11,-5), (-11,-4), (-11,-3), (-11,-2),
(-11,-1), (-11,0), (-11,1), (-11,2), (-11,3), (-11,4), (-11,5), (-11,6),
(-11,7), (-11,8), (-11,9), (-11,10), (-11,11), (-11,12), (-11,13), (-11,14),
(-10,-13), (-10,-12), (-10,-11), (-10,-10), (-10,-9), (-10,-8), (-10,-7),
(-10,-6), (-10,-5), (-10,-4), (-10,-3), (-10,-2), (-10,-1), (-10,0), (-10,1),
(-10,2), (-10,3), (-10,4), (-10,5), (-10,6), (-10,7), (-10,8), (-10,9),
(-10,10), (-10,11), (-10,12), (-10,13), (-10,14), (-9,-13), (-9,-12), (-9,-11),
(-9,-10), (-9,-9), (-9,-8), (-9,-7), (-9,-6), (-9,-5), (-9,-4), (-9,-3),
(-9,-2), (-9,-1), (-9,0), (-9,1), (-9,2), (-9,3), (-9,4), (-9,5), (-9,6),
(-9,7), (-9,8), (-9,9), (-9,10), (-9,11), (-9,12), (-9,13), (-9,14), (-8,-13),
(-8,-12), (-8,-11), (-8,-10), (-8,-9), (-8,-8), (-8,-7), (-8,-6), (-8,-5),
(-8,-4), (-8,-3), (-8,-2), (-8,-1), (-8,0), (-8,1), (-8,2), (-8,3), (-8,4),
(-8,5), (-8,6), (-8,7), (-8,8), (-8,9), (-8,10), (-8,11), (-8,12), (-8,13),
(-8,14), (-7,-13), (-7,-12), (-7,-11), (-7,-10), (-7,-9), (-7,-8), (-7,-7),
(-7,-6), (-7,-5), (-7,-4), (-7,-3), (-7,-2), (-7,-1), (-7,0), (-7,1), (-7,2),
(-7,3), (-7,4), (-7,5), (-7,6), (-7,7), (-7,8), (-7,9), (-7,10), (-7,11),
(-7,12), (-7,13), (-7,14), (-6,-13), (-6,-12), (-6,-11), (-6,-10), (-6,-9),
(-6,-8), (-6,-7), (-6,-6), (-6,-5), (-6,-4), (-6,-3), (-6,-2), (-6,-1), (-6,0),
(-6,1), (-6,2), (-6,3), (-6,4), (-6,5), (-6,6), (-6,7), (-6,8), (-6,9),
(-6,10), (-6,11), (-6,12), (-6,13), (-6,14), (-5,-13), (-5,-12), (-5,-11),
(-5,-10), (-5,-9), (-5,-8), (-5,-7), (-5,-6), (-5,-5), (-5,-4), (-5,-3),
(-5,-2), (-5,-1), (-5,0), (-5,1), (-5,2), (-5,3), (-5,4), (-5,5), (-5,6),
(-5,7), (-5,8), (-5,9), (-5,10), (-5,11), (-5,12), (-5,13), (-5,14), (-4,-13),
(-4,-12), (-4,-11), (-4,-10), (-4,-9), (-4,-8), (-4,-7), (-4,-6), (-4,-5),
(-4,-4), (-4,-3), (-4,-2), (-4,-1), (-4,0), (-4,1), (-4,2), (-4,3), (-4,4),
(-4,5), (-4,6), (-4,7), (-4,8), (-4,9), (-4,10), (-4,11), (-4,12), (-4,13),
(-4,14), (-3,-13), (-3,-12), (-3,-11), (-3,-10), (-3,-9), (-3,-8), (-3,-7),
(-3,-6), (-3,-5), (-3,-4), (-3,-3), (-3,-2), (-3,-1), (-3,0), (-3,1), (-3,2),
(-3,3), (-3,4), (-3,5), (-3,6), (-3,7), (-3,8), (-3,9), (-3,10), (-3,11),
(-3,12), (-3,13), (-3,14), (-2,-13), (-2,-12), (-2,-11), (-2,-10), (-2,-9),
(-2,-8), (-2,-7), (-2,-6), (-2,-5), (-2,-4), (-2,-3), (-2,-2), (-2,-1), (-2,0),
(-2,1), (-2,2), (-2,3), (-2,4), (-2,5), (-2,6), (-2,7), (-2,8), (-2,9),
(-2,10), (-2,11), (-2,12), (-2,13), (-2,14), (-1,-13), (-1,-12), (-1,-11),
(-1,-10), (-1,-9), (-1,-8), (-1,-7), (-1,-6), (-1,-5), (-1,-4), (-1,-3),
(-1,-2), (-1,-1), (-1,0), (-1,1), (-1,2), (-1,3), (-1,4), (-1,5), (-1,6),
(-1,7), (-1,8), (-1,9), (-1,10), (-1,11), (-1,12), (-1,13), (-1,14), (0,-13),
(0,-12), (0,-11), (0,-10), (0,-9), (0,-8), (0,-7), (0,-6), (0,-5), (0,-4),
(0,-3), (0,-2), (0,-1), (0,0), (0,1), (0,2), (0,3), (0,4), (0,5), (0,6), (0,7),
(0,8), (0,9), (0,10), (0,11), (0,12), (0,13), (0,14), (1,-13), (1,-12),
(1,-11), (1,-10), (1,-9), (1,-8), (1,-7), (1,-6), (1,-5), (1,-4), (1,-3),
(1,-2), (1,-1), (1,0), (1,1), (1,2), (1,3), (1,4), (1,5), (1,6), (1,7), (1,8),
(1,9), (1,10), (1,11), (1,12), (1,13), (1,14), (2,-13), (2,-12), (2,-11),
(2,-10), (2,-9), (2,-8), (2,-7), (2,-6), (2,-5), (2,-4), (2,-3), (2,-2),
(2,-1), (2,0), (2,1), (2,2), (2,3), (2,4), (2,5), (2,6), (2,7), (2,8), (2,9),
(2,10), (2,11), (2,12), (2,13), (2,14), (3,-13), (3,-12), (3,-11), (3,-10),
(3,-9), (3,-8), (3,-7), (3,-6), (3,-5), (3,-4), (3,-3), (3,-2), (3,-1), (3,0),
(3,1), (3,2), (3,3), (3,4), (3,5), (3,6), (3,7), (3,8), (3,9), (3,10), (3,11),
(3,12), (3,13), (3,14), (4,-13), (4,-12), (4,-11), (4,-10), (4,-9), (4,-8),
(4,-7), (4,-6), (4,-5), (4,-4), (4,-3), (4,-2), (4,-1), (4,0), (4,1), (4,2),
(4,3), (4,4), (4,5), (4,6), (4,7), (4,8), (4,9), (4,10), (4,11), (4,12),
(4,13), (4,14), (5,-13), (5,-12), (5,-11), (5,-10), (5,-9), (5,-8), (5,-7),
(5,-6), (5,-5), (5,-4), (5,-3), (5,-2), (5,-1), (5,0), (5,1), (5,2), (5,3),
(5,4), (5,5), (5,6), (5,7), (5,8), (5,9), (5,10), (5,11), (5,12), (5,13),
(5,14), (6,-13), (6,-12), (6,-11), (6,-10), (6,-9), (6,-8), (6,-7), (6,-6),
(6,-5), (6,-4), (6,-3), (6,-2), (6,-1), (6,0), (6,1), (6,2), (6,3), (6,4),
(6,5), (6,6), (6,7), (6,8), (6,9), (6,10), (6,11), (6,12), (6,13), (6,14),
(7,-13), (7,-12), (7,-11), (7,-10), (7,-9), (7,-8), (7,-7), (7,-6), (7,-5),
(7,-4), (7,-3), (7,-2), (7,-1), (7,0), (7,1), (7,2), (7,3), (7,4), (7,5),
(7,6), (7,7), (7,8), (7,9), (7,10), (7,11), (7,12), (7,13), (7,14), (8,-13),
(8,-12), (8,-11), (8,-10), (8,-9), (8,-8), (8,-7), (8,-6), (8,-5), (8,-4),
(8,-3), (8,-2), (8,-1), (8,0), (8,1), (8,2), (8,3), (8,4), (8,5), (8,6), (8,7),
(8,8), (8,9), (8,10), (8,11), (8,12), (8,13), (8,14), (9,-13), (9,-12),
(9,-11), (9,-10), (9,-9), (9,-8), (9,-7), (9,-6), (9,-5), (9,-4), (9,-3),
(9,-2), (9,-1), (9,0), (9,1), (9,2), (9,3), (9,4), (9,5), (9,6), (9,7), (9,8),
(9,9), (9,10), (9,11), (9,12), (9,13), (9,14), (10,-13), (10,-12), (10,-11),
(10,-10), (10,-9), (10,-8), (10,-7), (10,-6), (10,-5), (10,-4), (10,-3),
(10,-2), (10,-1), (10,0), (10,1), (10,2), (10,3), (10,4), (10,5), (10,6),
(10,7), (10,8), (10,9), (10,10), (10,11), (10,12), (10,13), (10,14), (11,-13),
(11,-12), (11,-11), (11,-10), (11,-9), (11,-8), (11,-7), (11,-6), (11,-5),
(11,-4), (11,-3), (11,-2), (11,-1), (11,0), (11,1), (11,2), (11,3), (11,4),
(11,5), (11,6), (11,7), (11,8), (11,9), (11,10), (11,11), (11,12), (11,13),
(11,14), (12,-13), (12,-12), (12,-11), (12,-10), (12,-9), (12,-8), (12,-7),
(12,-6), (12,-5), (12,-4), (12,-3), (12,-2), (12,-1), (12,0), (12,1), (12,2),
(12,3), (12,4), (12,5), (12,6), (12,7), (12,8), (12,9), (12,10), (12,11),
(12,12), (12,13), (12,14), (13,-13), (13,-12), (13,-11), (13,-10), (13,-9),
(13,-8), (13,-7), (13,-6), (13,-5), (13,-4), (13,-3), (13,-2), (13,-1), (13,0),
(13,1), (13,2), (13,3), (13,4), (13,5), (13,6), (13,7), (13,8), (13,9),
(13,10), (13,11), (13,12), (13,13), (13,14), (14,-13), (14,-12), (14,-11),
(14,-10), (14,-9), (14,-8), (14,-7), (14,-6), (14,-5), (14,-4), (14,-3),
(14,-2), (14,-1), (14,0), (14,1), (14,2), (14,3), (14,4), (14,5), (14,6),
(14,7), (14,8), (14,9), (14,10), (14,11), (14,12), (14,13), (14,14)]
