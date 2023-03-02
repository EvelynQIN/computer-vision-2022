import numpy as np
from scipy import signal, ndimage
import cv2

# Harris corner detector
def extract_harris(img, sigma = 1.0, k = 0.05, thresh = 1e-5):
    '''
    Inputs:
    - img:      (h, w) gray-scaled image
    - sigma:    smoothing Gaussian sigma. suggested values: 0.5, 1.0, 2.0
    - k:        Harris response function constant. suggest interval: (0.04 - 0.06)
    - thresh:   scalar value to threshold corner strength. suggested interval: (1e-6 - 1e-4)
    Returns:
    - corners:  (q, 2) numpy array storing the keypoint positions [x, y]
    - C:     (h, w) numpy array storing the corner strength (harris reponse)
    '''
    # Convert to float
    img = img.astype(float) / 255.0

    # Compute image gradients
    # TODO: implement the computation of the image gradients Ix and Iy here.
    # You may refer to scipy.signal.convolve2d for the convolution.
    # Do not forget to use the mode "same" to keep the image size unchanged.
 
    Ix = signal.convolve2d(img, [[0.5, 0, -0.5]], boundary = 'symm', mode = 'same')
    Iy = signal.convolve2d(img, [[0.5], [0], [-0.5]], boundary = 'symm', mode = 'same')

    # Compute local auto-correlation matrix
    # TODO: compute the auto-correlation matrix here
    # You may refer to cv2.GaussianBlur for the gaussian filtering (border_type=cv2.BORDER_REPLICATE)
    Ixx = cv2.GaussianBlur(Ix * Ix, ksize = (0, 0), sigmaX = sigma, sigmaY = sigma, borderType=cv2.BORDER_REPLICATE) 
    Iyy = cv2.GaussianBlur(Iy * Iy, ksize = (0, 0), sigmaX = sigma, sigmaY = sigma, borderType=cv2.BORDER_REPLICATE)
    Ixy = cv2.GaussianBlur(Ix * Iy, ksize = (0, 0), sigmaX = sigma, sigmaY = sigma, borderType=cv2.BORDER_REPLICATE) 

    # Compute Harris response function
    # TODO: compute the Harris response function C here
    det = Ixx * Iyy - Ixy ** 2
    trace = Ixx + Iyy
    C = det - k * trace ** 2

    # Detection with threshold
    # TODO: detection and find the corners here
    # For the local maximum check, you may refer to scipy.ndimage.maximum_filter to check a 3x3 neighborhood.
    local_max = ndimage.maximum_filter(C, size = 3)
    corners = np.argwhere(np.logical_and(C > thresh, C == local_max))
    return np.stack((corners[:, 1], corners[:, 0]), axis = -1), C # inverse the coordinates of corners for openCV to plot the points

 