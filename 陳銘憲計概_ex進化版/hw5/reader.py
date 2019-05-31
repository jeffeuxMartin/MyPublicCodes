'''
DO NOT MODIFY THIS FILE 
'''
def load_fashion_mnist(path, kind='train'):
    import os
    import numpy as np

    """Load fashion MNIST data from `path`"""

    labels = []

    if kind == 'train':
        labels_path = os.path.join(path,'%s-labels-idx1-ubyte.npz'% kind)
        labels = np.load(labels_path)['y']

    images_path = os.path.join(path,'%s-images-idx3-ubyte.npz'% kind)
    images = np.load(images_path)['x']

    return images, labels