import matplotlib.pyplot as plt
from .cupy_or_numpy import xp
from .decorators import get_jit_decorator, get_decorator

from .gradient import gradient
from .partials import partials
from .make_k import make_k9, make_k11
from .gcsrm import gcsrm_1D_v2

@get_jit_decorator()
def iterate_gradients(x, dx, K1, K2, K3, B, dims, bsize, u, A, T, st, des):
    db, du = xp.zeros(bsize), 0
    Dtdb = xp.zeros((bsize, st.size, st.size))
    Dtdu = xp.zeros((st.size+1, st.size+1))
    spikes = xp.sort(xp.concatenate((st,des)))
    for i in range(spikes.size):
        t = spikes[i]
        tO = st[st<=t]
        if t == tO[-1]:
            partials(x, dx, K1, K2, K3, B, dims, tO, u, A, Dtdb, Dtdu)
        ddb, ddu = gradient(tO, des[ des<=t ], t+1, T, Dtdb, Dtdu)
        db += ddb
        du += ddu
    return st, db, du

@get_decorator()
def sim_1D(x, dx, B, b, dims, u, T, A, des):
    K = make_k9(B,b) #smoothness included from a global
    # print([i.shape for i in K[1:]])
    c, y, st = gcsrm_1D_v2(x, K, u, A)

    bsize = xp.sum([i.size for i in b[1:]])
    if des.size > 0:
        return iterate_gradients(x, dx, K[1], K[2], K[3], B, dims, bsize, u, A, T, st, des)
    else:
        return st, xp.zeros(bsize), 0
