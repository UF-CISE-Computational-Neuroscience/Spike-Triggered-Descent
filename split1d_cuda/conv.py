from .cupy_or_numpy import xp
# import cupy as cp
from .decorators import get_jit_decorator, get_decorator
from .constants import smoothness
from numba import cuda

@get_jit_decorator()
def nbconv_1d2o(k, x1, x2): #numba_conv
    l = x1.size - k.shape[0] + 1
    y = xp.zeros(l)
    for n in range(l):
        d = n+k.shape[0]-1
        for i in range(k.shape[0]):
            for j in range(k.shape[1]):
                y[n] += x1[d-i] * x2[d-j] * k[i,j]
    return y

# @get_jit_decorator()
@cuda.jit
def nbconv_1d2o_v2(x, k, y): #numba_conv
    n = cuda.grid(1)
    b_offset = smoothness*3 - 1
    if (0 <= n) and (n < y.size):
        d = n+k.shape[0]-1
        for i in range(k.shape[0]):
            for j in range(min(i+b_offset, k.shape[1])):
                y[n] += x[d-i] * x[d-j] * k[i,j]

@get_jit_decorator()
def nbconv_1d3o_v1(k, x1, x2, x3): #numba_conv
    l = x1.size - k.shape[0] + 1
    y = xp.zeros(l)
    for n in range(l):
        d = n+k.shape[0]-1
        for i in range(k.shape[0]):
            for j in range(k.shape[1]):
                for l in range(k.shape[2]):
                    y[n] += x1[d-i] * x2[d-j] * x3[d-l] * k[i,j,l]
    return y

@get_jit_decorator()
def nbconv_1d3o_v2(k, x1, x2, x3):
    #convolve only the cone
    l = x1.size - k.shape[0] + 1
    y = xp.zeros(l)
    # make sure to catch the tail of the spline, beta smoothness is usually 4
    # match in setup_betas
    # smoothness * 3 - 1 = 2, 5, 8, 11, 14
    # b_offset = 11
    b_offset = smoothness*3 - 1
    for n in range(l):
        d = n+k.shape[0]-1
        for i in range(k.shape[0]):
            for j in range(min(i+b_offset, k.shape[1])):
                for l in range(min(j+b_offset, k.shape[2])):
                    y[n] += x1[d-i] * x2[d-j] * x3[d-l] * k[i,j,l]
    return y


# @get_jit_decorator()
@cuda.jit
def nbconv_1d3o_v3_cuda(x,k,y):
    #convolve only the cone
    # make sure to catch the tail of the spline, beta smoothness is usually 4
    # match in setup_betas
    # smoothness * 3 - 1 = 2, 5, 8, 11, 14
    # b_offset = 11
    b_offset = smoothness*3 - 1
    n = cuda.grid(1)
    if (0 <= n) and (n < y.size):
        d = n+k.shape[0]-1
        for i in range(k.shape[0]):
            for j in range(min(i+b_offset, k.shape[1])):
                for l in range(min(j+b_offset, k.shape[2])):
                    y[n] += x[d-i] * x[d-j] * x[d-l] * k[i,j,l]

# @get_jit_decorator()
# @cuda.jit
def nbconv_1d3o_v3(x, k, y):
    # l = x.size - k.shape[0] + 1
    # y = xp.zeros(l)
    xc = cuda.to_device(x)
    kc = cuda.to_device(k)
    # yc = cuda.to_device(y)
    th = 128
    b = y.size//th+1
    # print(b,th)
    nbconv_1d3o_v3_cuda[b,th](xc, kc, y)
    # return y

@get_jit_decorator()
def nbconv_1d1o(k, x): #numba_conv
    l = x.size - k.shape[0] + 1
    y = xp.zeros(l)
    for n in range(l):
        d = n+k.shape[0]-1
        for i in range(k.shape[0]):
            y[n] += x[d-i] * k[i]
    return y

# @get_jit_decorator()
@cuda.jit
def nbconv_1d1o_v2(x, k, y): #numba_conv
    n = cuda.grid(1)
    if (0 <= n) and (n < y.size):
        d = n+k.shape[0]-1
        for i in range(k.shape[0]):
            y[n] += x[d-i] * k[i]

@get_decorator()
def conv_1d1o(k, x):
    return xp.convolve(x, k, 'valid')
