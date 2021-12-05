from .cupy_or_numpy import xp
from .decorators import get_jit_decorator, get_decorator
from .conv import nbconv_1d1o, nbconv_1d1o_v2, nbconv_1d2o, nbconv_1d2o_v2, nbconv_1d3o_v1, nbconv_1d3o_v2, nbconv_1d3o_v3, nbconv_1d3o_v3_cuda
from .get_spikes import get_spikes_2
from numba import cuda, njit, jit
import numpy as np

# @get_jit_decorator()
# @jit
def get_c(x,K):
    xc = cuda.to_device(x)
    k1 = cuda.to_device(K[1])
    k2 = cuda.to_device(K[2])
    k3 = cuda.to_device(K[3])
    th = 128
    b = x.size//th+1
    c = cuda.device_array_like(x)

    # c = xp.zeros( x.size )
    if 0 < K[1].shape[0]:
        c[K[1].shape[0]-1:] += nbconv_1d1o(K[1], x)
        nbconv_1d1o_v2[b,th](xc, k1, c[K[1].shape[0]-1:])
    if 0 < K[2].shape[0]:
        # c[K[2].shape[0]-1:] += nbconv_1d2o(K[2], x, x)
        nbconv_1d2o_v2[b,th](xc, k2, c[K[2].shape[0]-1:])
    if 0 < K[3].shape[0]:
        # c[K[3].shape[0]-1:] += nbconv_1d3o_v2(K[3], x, x, x)
        # l = x.size - K[3].shape[0] + 1
        # y = xp.zeros(l)
        # xc = cuda.to_device(x)
        # kc = cuda.to_device(K[3])
        # c[K[3].shape[0]-1:] += nbconv_1d3o_v3(xc, kc)
        # c[K[3].shape[0]-1:] += nbconv_1d3o_v3(x, K[3])
        # nbconv_1d3o_v3(x, k3, c[K[3].shape[0]-1:])
        nbconv_1d3o_v3_cuda[b,th](xc, k3, c[K[3].shape[0]-1:])
    return c.copy_to_host()
    # return c

# @cuda.jit
# @njit
def get_c_v2(x,K1,K2,K3,c,b,th):
    if 0 < K1.shape[0]:
        nbconv_1d1o_v2[b,th](x, K1, c[K1.shape[0]-1:])
    if 0 < K2.shape[0]:
        nbconv_1d2o_v2[b,th](x, K2, c[K2.shape[0]-1:])
    if 0 < K3.shape[0]:
        nbconv_1d3o_v3_cuda[b,th](x, K3, c[K3.shape[0]-1:])

# @get_decorator()
# @njit
def get_c_helper(x, K):
    xc = cuda.to_device(x)
    K1 = cuda.to_device(K[1])
    K2 = cuda.to_device(K[2])
    K3 = cuda.to_device(K[3])
    th = 128
    b = x.size//th+1
    c = cuda.device_array_like(x)
    get_c_v2(xc,K1,K2,K3,c,b,th)
    return c.copy_to_host()

# @get_decorator()
# @njit
try:
    import cusignal as cs
    import cupy as cp
    def volterra(x, K):
        c = cp.zeros(x.size)
        if 0 < K[1].shape[0]:
            c[K[1].shape[0]-1:] += cs.convolve(x, K[1], 'valid')
        if 0 < K[2].shape[0]:
            c[K[2].shape[0]-1:] += cs.convolve1d2o(x, K[2])
        if 0 < K[3].shape[0]:
            c[K[3].shape[0]-1:] += cs.convolve1d3o(x, K[3])
        return cp.asnumpy(c)
except:
    print('using non cusignal version')
    def volterra(x, K):
        c = xp.zeros(x.size)
        if 0 < K[1].shape[0]:
            c[K[1].shape[0]-1:] += xp.convolve(x, K[1], 'valid')
        if 0 < K[2].shape[0]:
            c[K[2].shape[0]-1:] += nbconv_1d2o(K[2], x, x)
        if 0 < K[3].shape[0]:
            c[K[3].shape[0]-1:] += nbconv_1d3o_v2(K[3], x, x, x)
        return c

# @get_decorator()
def gcsrm_1D_v2(x, K, u, A):
    c = volterra(x, K)
    s = max( [ i.shape[0] for i in K[1:] ] )
    c = c[s-1:]
    st, y = get_spikes_2(c, K[0], u, A)

    return c, y, st + (x.size - c.size) #correct spike times