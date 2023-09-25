
import numpy as np
cimport numpy as cnp
cimport cython
from libc.math cimport sqrt, ceil
cnp.import_array()

ctypedef fused np_floats:
    cnp.float32_t
    cnp.float64_t


cdef inline Py_ssize_t round(np_floats r) nogil:
    return <Py_ssize_t>(
        (r + <np_floats>0.5) if (r > <np_floats>0.0) else (r - <np_floats>0.5)
    )


def _weighted_hough_line(cnp.ndarray[ndim=2, dtype=cnp.uint64_t]  img,
                cnp.ndarray[ndim=1, dtype=cnp.float64_t] theta):
                
    

    # Compute the array of angles and their sine and cosine
    cdef cnp.ndarray[ndim=1, dtype=cnp.float64_t] ctheta
    cdef cnp.ndarray[ndim=1, dtype=cnp.float64_t] stheta

    ctheta = np.cos(theta)
    stheta = np.sin(theta)

    # compute the bins and allocate the accumulator array
    cdef cnp.ndarray[ndim=2, dtype=cnp.uint64_t] accum
    cdef cnp.ndarray[ndim=1, dtype=cnp.float64_t] bins
    cdef Py_ssize_t max_distance, offset
    
    #image diagonal
    offset = <Py_ssize_t>ceil(sqrt(img.shape[0] * img.shape[0] +
                                   img.shape[1] * img.shape[1]))
    max_distance = 2 * offset + 1
    accum = np.zeros((max_distance, theta.shape[0]), dtype=np.uint64)
    bins = np.linspace(-offset, offset, max_distance)

    # compute the nonzero indexes
    cdef cnp.ndarray[ndim=1, dtype=cnp.npy_intp] x_idxs, y_idxs
    y_idxs, x_idxs = np.nonzero(img)

    # finally, run the transform
    cdef Py_ssize_t nidxs, nthetas, i, j, x, y, accum_idx, weight

    nidxs = y_idxs.shape[0]  # x and y are the same shape
    nthetas = theta.shape[0]
    cdef cnp.ndarray[ndim=2, dtype=cnp.uint64_t]  mv = img

    with nogil:
        for i in range(nidxs):
            x = x_idxs[i]
            y = y_idxs[i]
            weight = mv[y,x]

            for j in range(nthetas):
                accum_idx = round((ctheta[j] * x + stheta[j] * y)) + offset
                accum[accum_idx, j] += weight

    return accum, theta, bins



def _sliding_window_hough_line(cnp.ndarray[ndim=2, dtype=cnp.uint64_t]  img,
                cnp.ndarray[ndim=1, dtype=cnp.float64_t] theta,
                cython.bint vertical = True,
                Py_ssize_t window_div = 5):

    cdef cnp.ndarray[ndim=2, dtype=cnp.uint64_t]  mv, accum, temp_accum
    cdef cnp.ndarray[ndim=1, dtype=cnp.float64_t] bins
    cdef Py_ssize_t  w_size, stride, axis, i, accum_size

    accum_size = <Py_ssize_t>ceil(sqrt(img.shape[0] * img.shape[0] +
                                   img.shape[1] * img.shape[1]))*2+1
    accum = np.zeros((accum_size, theta.shape[0]), dtype=np.uint64)
    mv =  np.zeros_like(img)#, dtype=np.uint64)

    # define the stride. because the sub-windows will overlap, it is defined as l / window_div * 3/2, l being either height or width
    if vertical:
        axis = img.shape[0]
        w_size = round(axis/window_div)
        stride = round(window_div / 2)
        

        for i in (-1,1):
            mv.fill(0)
            mv[:i*stride, :] = img[:i*stride, :]
            temp_accum, theta, bins = _weighted_hough_line(mv, theta)
            accum = accum+temp_accum

        for i in range(window_div*2-1):
            mv.fill(0)
            mv[i*stride:(i+1)*stride, :] = img[i*stride:(i+1)*stride, :]
            temp_accum, theta, bins = _weighted_hough_line(mv, theta)
            accum = accum+temp_accum

    else:
        axis = img.shape[1]
        w_size = round(axis/window_div)
        stride = round(window_div / 2)

        for i in (-1,1):
            mv.fill(0)
            mv[:, :i*stride] = img[:, :i*stride]
            temp_accum, theta, bins = _weighted_hough_line(mv, theta)
            accum = accum+temp_accum

        for i in range(window_div*2-1):
            mv.fill(0)
            mv[:, i*stride:(i+1)*stride] = img[:, i*stride:(i+1)*stride]
            temp_accum, theta, bins = _weighted_hough_line(mv, theta)
            accum = accum+temp_accum
    
    return accum, theta, bins
    
def _CC_size_weight(
    cnp.ndarray[ndim=2, dtype=cnp.uint8_t]  img,
    cython.bint filter = True):
    
    cdef cnp.ndarray[ndim=2, dtype=cnp.uint64_t]  mv