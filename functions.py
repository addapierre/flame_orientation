"""
useful functions for flame detection, including modified hough line algorithm in cython
"""

import numpy as np
import cv2
import pyximport; pyximport.install()
from _weighted_hough import _weighted_hough_line, _sliding_window_hough_line


def image_preprocessing(img):
    """
    skeletonizes a grayscale image
    """
    if img.shape[2]==3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    sobely = cv2.Sobel(src=img, ddepth=cv2.CV_8U, dx=0, dy=1, ksize=5) # suppress vertical lines

    ksize = 11
    sobel_blur = cv2.GaussianBlur(sobely, (ksize,ksize), 0) # blur to make small texture disappear
    _,thresh = cv2.threshold(sobel_blur,127,255,cv2.THRESH_BINARY)

    # gabor kernel for skeletonisation
    ks = 5; sigma = 3; theta = 90*(np.pi/180); lambd = 4; gamma=1; psi=np.pi*0.5
    g_kernel = cv2.getGaborKernel((ks, ks), sigma, theta, lambd,gamma, psi, cv2.CV_32F )
    return cv2.filter2D(thresh, cv2.CV_8UC3, g_kernel)


def CC_img(img, filter = True):

    """
    assign weights to pixel in a binary image, according to the pixel blob size, using a connected component algorithm.

    - img: binary skeletonized image
    - filter (bool): if set to True, removes all pixels with a value lower than the median of non-zero pixels.
    """
    if len(img.shape) == 3:
        stats = cv2.connectedComponentsWithStats(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), connectivity=8)
    else:
        stats = cv2.connectedComponentsWithStats(img, connectivity=8)
    new_img = np.zeros(stats[1].shape)
    for i in range(1, stats[0]):
        new_img[np.where(stats[1]==i)] = stats[2][i,4]
    new_img = np.sqrt(new_img)

    if filter: 
        #remove small blobs
        x, y = np.nonzero(new_img)
        median = np.median(new_img[x, y])
        new_img[new_img<median]=0
    return new_img


def weighted_hough_line(img, theta = None):

    """Perform a straight line Hough transform with weighted pixels. 
    The Hough space accumulator is incremented based on the pixel values. 
    If all pixels are set to 1, the normal Hough line algorithm is performed.

    Parameters
    ----------
    img : (M, N) ndarray
        Input image with nonzero values representing edges. 
    theta (optional) : 1D array  
        array of angles to be considered for line detection, in radians. 
        If None, consider all angles in the range 0 to pi 
        NB: 0 is vertical
    Returns
    -------
    H : 2-D ndarray of uint64
        Hough transform accumulator.
    theta : ndarray
        Angles at which the transform was computed, in radians.
    distances : ndarray
        Distance values.

    Notes
    -----
    The origin is the top left corner of the original image.
    X and Y axis are horizontal and vertical edges respectively.
    The distance is the minimal algebraic distance from the origin
    to the detected line.
    """
    assert len(img.shape) == 2, f"the image should be binary, instead got an image of shape {img.shape}"

    if theta is None:
        theta = np.linspace(0,np.pi, 180, dtype = np.float64) 

    return _weighted_hough_line(img.astype(np.uint64), theta)


def sliding_window_hough_line(img, theta = None, vertical = True, window_div = 5):
    """Perform a straight line Hough transform with weighted pixels. 
    The Hough space accumulator is incremented based on the pixel values. 
    If all pixels are set to 1, the normal Hough line algorithm is performed.
    the image is divided along the vertical or horizontal axis in a number of subpart (default 5).
    The hough line algorithm is applied to each subpart. This allows to limit the hough line algorithm's scope in the vertical or horizontal axis.

    Parameters
    ----------
    img : (M, N) ndarray
        Input image with nonzero values representing edges.
    theta (optional) : 1D array  
        array of angles to be considered for line detection, in radians. 
        If None, consider all angles in the range 0 to pi 
        NB: 0 is vertical
    vertical (boolean, default = True):  
        determines whether the image will be divided along the vertical or horizontal axis.
    window_div (int, default = 5):
        determines in how many subpart the image should be divided, along the axis specified by the "vertical" parameter.

    Returns
    -------
    H : 2-D ndarray of uint64
        Hough transform accumulator.
    theta : ndarray
        Angles at which the transform was computed, in radians.
    distances : ndarray
        Distance values.
    """
    assert len(img.shape) == 2, f"the image should be binary, instead got an image of shape {img.shape}"

    if theta is None:
        theta = np.linspace(0,np.pi, 180, dtype = np.float64) 

    return _sliding_window_hough_line(img.astype(np.uint64), theta, vertical, window_div)