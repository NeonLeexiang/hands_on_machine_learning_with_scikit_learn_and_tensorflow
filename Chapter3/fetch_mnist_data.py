"""
    date:       2019/12/30 3:48 下午
    written by: neonleexiang
"""


# import list
import os
import numpy as np
from scipy.io import loadmat
import matplotlib
import matplotlib.pyplot as plt


"""
    Warning: fetch_mldata() is deprecated since Scikit-Learn 0.20. 
    You should use fetch_openml() instead. 
    However, it returns the unsorted MNIST dataset, 
    whereas fetch_mldata() returned the dataset sorted by target 
    (the training set and the test test were sorted separately). 
    In general, this is fine, but if you want to get the exact same results as before, 
    you need to sort the dataset using the following function:

"""


# need to sort the data if import the data from openml
def sort_by_target(mnist):
    reorder_train = np.array(sorted([(target, i) for i, target in enumerate(mnist.target[:60000])]))[:, 1]
    reorder_test = np.array(sorted([(target, i) for i, target in enumerate(mnist.target[60000:])]))[:, 1]
    mnist.data[:60000] = mnist.data[reorder_train]
    mnist.target[:60000] = mnist.target[reorder_train]
    mnist.data[60000:] = mnist.data[reorder_test + 60000]
    mnist.target[60000:] = mnist.target[reorder_test + 60000]


def sklearn_import_data():
    """
    there is some difference between sklearn < 0.20 and > 0.20
    :return: data, target
    """
    try:
        from sklearn.datasets import fetch_openml
        mnist = fetch_openml('mnist_784', version=1, cache=True)
        mnist.target = mnist.target.astype(np.int8)  # fetch_openml() returns targets as strings
        sort_by_target(mnist)  # fetch_openml() returns an unsorted dataset
    except ImportError:
        from sklearn.datasets import fetch_mldata
        mnist = fetch_mldata('MNIST original')
    return mnist["data"], mnist["target"]


"""
    we found that it occur URLError and here is the detail:
    urllib.error.URLError: <urlopen error [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed (_ssl.c:749)>
"""
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


"""
    we found that it is too slow to download the data from web
    so we have the data mnist.mat in data set,
    now we use scipy.io to fetch the data
"""


MNIST_PATH = "datasets/mnist"


def load_data_from_datasets(mnist_path=MNIST_PATH):
    # mat_path = os.path.join(mnist_path, 'mnist-original.mat')
    # mnist = loadmat(mat_path)
    # # mnist.target = mnist.target.astype(np.int8)  # fetch_openml() returns targets as strings
    # # sort_by_target(mnist)  # fetch_openml() returns an unsorted dataset
    # return np.array(mnist['data']).reshape([70000, 784]), np.array(mnist['label']).reshape([70000, ])
    from sklearn.datasets import fetch_mldata
    mnist = fetch_mldata('mnist-original', data_home='./datasets')
    return mnist['data'], mnist['target']


def plot_mnist_digit(data):
    digit = data.reshape(28, 28)

    plt.imshow(digit,
               cmap=matplotlib.cm.binary,
               interpolation='nearest')
    plt.axis('off')
    plt.show()


# EXTRA
def plot_digits(instances, images_per_row=10, **options):
    size = 28
    images_per_row = min(len(instances), images_per_row)
    images = [instance.reshape(size,size) for instance in instances]
    n_rows = (len(instances) - 1) // images_per_row + 1
    row_images = []
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((size, size * n_empty)))
    for row in range(n_rows):
        rimages = images[row * images_per_row : (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    image = np.concatenate(row_images, axis=0)
    plt.imshow(image, cmap = matplotlib.cm.binary, **options)
    plt.axis("off")


def plot_some_digits(X):
    plt.figure(figsize=(9, 9))
    example_images = np.r_[X[:12000:600], X[13000:30600:600], X[30600:60000:590]]
    plot_digits(example_images, images_per_row=10)
    plt.show()


if __name__ == '__main__':
    # X, y = sklearn_import_data()
    # print(X, y)
    X, y = load_data_from_datasets()
    # print(X, y)
    # print(X[36000], y[36000])
    # plot_mnist_digit(X[36000])
    # print(X.shape, y.shape, X.dtype, y.dtype)
    plot_some_digits(X)
