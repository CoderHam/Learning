cdef extern from "math.h":
    double sqrt(double x)

cdef extern from "math.h":
    double pow(double x, double y)

cimport numpy as np
from numpy cimport ndarray
cimport cython

@cython.boundscheck(False)
def rmse(ndarray[np.float64_t, ndim=1] a not None,ndarray[np.float64_t, ndim=1] b not None):
    cdef Py_ssize_t i
    cdef Py_ssize_t n = a.shape[0]
    cdef double m = 0.0
    for i in range(n):
        m += pow(a[i]-b[i],2.0)
    return sqrt(m/n)
