import time
import os
import random
import math
import torch
import numpy as np

# run `pip install scikit-image` to install skimage, if you haven't done so
from skimage import io, color
from skimage.transform import rescale

def distance(x, X):
    """
    :param x: 3-dim tensor representing the CIELAB value of one pixel
    :param X: Nx3 matrix containing CIELAB values of every pixel
    :return:
        dist: N-dim tensor containing for each pixel in X the distance to the pixel x
    """
    dist = torch.norm(X - x, dim = 1)
    return dist

def distance_batch(x, X):
    """
    :param x: Bs x 3
    :patam X: N x 3
    :return:
        dist: Bs x N
    """
    dist = torch.cdist(x, X, p = 2) # (Bs, N)
    return dist

def gaussian(dist, bandwidth):
    """
    :param dist: N-dim tensor
    :param bandwidth: the bandwidth of the Guassian kernel
    :return:
        weight: N-dim tensor containing weights of every point as a Gaussian kernel function of dist
    """
    weight = torch.exp(- (dist ** 2) / (2 * bandwidth ** 2))
    return weight

def update_point(weight, X):
    """
    :param weight: N-dim tensor containing weights of every point
    :param X: Nx3 matrix containing CIELAB values of every pixel
    :return:
        updated_point: 3-dim vector containing the weighted average of all points
    """
    weight = weight / torch.sum(weight) # ensure all the weights sum to 1
    updated_point = torch.matmul(weight.reshape(1, -1), X) # compute the weighted average of all points w^T * X
    return updated_point


def update_point_batch(weight, X):
    """
    :param weight: Bs x N
    :patam X: N x 3
    :return:
        updated_point: Bs x 3
    """
    weight = torch.div(weight, torch.sum(weight, dim = 1).reshape(-1, 1)) # ensure all the weights sum to 1
    updated_point = torch.matmul(weight, X)
    return updated_point

def meanshift_step(X, bandwidth=2.5):
    X_ = X.clone()
    for i, x in enumerate(X):
        dist = distance(x, X)
        weight = gaussian(dist, bandwidth)
        X_[i] = update_point(weight, X)
    return X_

def meanshift_step_batch(X, bandwidth=2.5):
    X_ = X.clone()
    batch_size = X.shape[0]
    for i in range(0, X.shape[0], batch_size):
        dist = distance_batch(X[i : i + batch_size], X)
        weight = gaussian(dist, bandwidth)
        X_[i : i + batch_size] = update_point_batch(weight, X)
    return X_


def meanshift(X):
    X = X.clone()
    for _ in range(20):
        # X = meanshift_step(X)   # slow implementation
        X = meanshift_step_batch(X, bandwidth = 2.5)   # fast implementation
    return X

scale = 0.25    # downscale the image to run faster

# Load image and convert it to CIELAB space
image = rescale(io.imread('cow.jpg'), scale, multichannel=True)
image_lab = color.rgb2lab(image)
shape = image_lab.shape # record image shape (49, 75, 3)
image_lab = image_lab.reshape([-1, 3])  # flatten the image (3675, 3)

# Run your mean-shift algorithm
t = time.time()
X = meanshift(torch.from_numpy(image_lab)).detach().cpu().numpy()
# X = meanshift(torch.from_numpy(data).cuda()).detach().cpu().numpy()  # you can use GPU if you have one
t = time.time() - t
print ('Elapsed time for mean-shift: {}'.format(t))

# Load label colors and draw labels as an image
colors = np.load('colors.npz')['colors'] # (24, 3)
colors[colors > 1.0] = 1
colors[colors < 0.0] = 0

centroids, labels = np.unique((X / 4).round(), return_inverse=True, axis=0)

# print the number of labels deteted
print("The number of detected labels are : {}".format(centroids.shape[0]))

result_image = colors[labels].reshape(shape)
result_image = rescale(result_image, 1 / scale, order=0, multichannel=True)     # resize result image to original resolution (order = 1 is 1NN)
result_image = (result_image * 255).astype(np.uint8)
io.imsave('result.png', result_image)
