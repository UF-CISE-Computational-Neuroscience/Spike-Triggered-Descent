import matplotlib.pyplot as plt
from .cupy_or_numpy import xp
from .decorators import get_jit_decorator, get_decorator
from .basis_np import Basis as basis_np
from .learning import mask_diagonal_iband, mask_diagonal_iband_3, make_b, make_b2, update
from .sim import sim_1D
from .constants import smoothness
from .E import E

from .make_k import make_k9, make_k11

@get_decorator()
def window(a,i,s):
    ai = a[ xp.logical_and( i < a, a < i + s) ]
    return ai - i

@get_decorator()
def Ecrawl_des(a, b, s, T):
    y = xp.zeros( a.size )
    for i in range(0,a.size):
        ai = window(a,a[i],s)
        bi = window(b,a[i],s)
        y[i] = E(ai, bi, 150)
    # return xp.trapz(y)
    return xp.sum(y)


@get_decorator()
def setup_betas(dims):
    (k, B, b) = basis_np( max(dims), smoothness )
    s = b.size
    m = 25 # multiplier

    b_des = make_b2(dims,m)
    b_lrn = make_b2(dims,m)
    mask2 = mask_diagonal_iband(dims[1])
    mask3 = mask_diagonal_iband_3(dims[2])
    b_des[2] *= mask2
    b_lrn[2] *= mask2
    b_des[3] *= mask3
    b_lrn[3] *= mask3

    masks = [1, mask2, mask3]
    return B, b_des, b_lrn, masks

@get_decorator()
def dist(b_des, b_lrn):
    return xp.linalg.norm(b_lrn - b_des) / xp.linalg.norm(b_des)

@get_decorator()
# @get_jit_decorator()
def learn(epoch, dims, b_lrn, db, masks):
    start, end = 0, 0
    for i in range(len(dims)):
        if 0 < dims[i]:
            start = end
            end += dims[i]**(i+1)
            b_lrn[i+1] += db[start:end] * masks[i]


def nearests_normed(st_lrn, st_des):
    if st_lrn.size == 0:
        return 10
    else:
        a = xp.zeros(st_lrn.size)
        for i in range(a.size):
            a[i] = xp.min( xp.abs(st_des - st_lrn[i]) )
        return xp.average(a)
    # return xp.sum(xp.clip(a, 0, 2*xp.average(a)))/a.size

def common_range(st_lrn, st_des):
    m, M = max(st_lrn[0], st_des[0]), min(st_lrn[-1], st_des[-1])
    st_lrn = st_lrn[ m < st_lrn ]
    st_lrn = st_lrn[ st_lrn < M ]
    st_des = st_des[ m < st_des ]
    st_des = st_des[ st_des < M ]
    return st_lrn, st_des

def nearests_normed2(st1, st2):
    a = xp.zeros(st1.size)
    for i in range(a.size):
        a[i] = xp.min( xp.abs(st2 - st1[i]) )
    a = xp.sort(a)
    # return xp.average( a[:int(a.size*0.8)] )
    return a

@get_decorator()
def record(epoch, ep_skip, U, distances, dims, b_des, b_lrn, u, st_des, st_lrn):
    if epoch%ep_skip == 0:
        ep = epoch//ep_skip
        U[ep] = u
        for i in range(3):
            if 0 < dims[i]:
                distances[i,ep] = dist(b_des[i+1], b_lrn[i+1])
                # distances[i,ep] = xp.linalg.norm(b_lrn[i+1])
        printable = "{0:.4f}  {1:.4f}  {2:.4f}  {3:.4f}".format(U[ep], distances[0,ep], distances[1,ep], distances[2,ep])
        # print(epoch, len(st_des), len(st_lrn), printable, E(st_des, st_lrn, 150))

        print(epoch, len(st_des), len(st_lrn), printable) #, nearests_normed(st_lrn, st_des))

def simulate(x, dx, segment_length, B, b_lrn, dims, u_lrn, T, A, st_des, b):
    left = xp.random.randint(0, x.size - segment_length)
    right = left + segment_length
    xi = x[left:right]
    dxi = dx[left:right]

    # ignore spikes without support
    des = st_des[ xp.argwhere(xp.logical_and( B[0].size+left<st_des, st_des<right )) ].flatten() - left
    # return sim_1D(xi, dxi, B, b_lrn, dims, u_lrn, T, des)
 
    if b:
        st_lrn, db_th, du_th = sim_1D(xi, dxi, B, b_lrn, dims, u_lrn, T, A, des)
    else:
        st_lrn, db_th, du_th = sim_1D(xi, dxi, B, b_lrn, dims, u_lrn, T, A, xp.empty( shape=(0), dtype=xp.float64 ) )

    return st_lrn, db_th, du_th, des

@get_decorator()
def learn_segments(x, st_des, dims, b_lrn, u_lrn, T, A, epochs, ep_skip, spikes_per_epoch, time_scale, p_lr, caps):
    B, b_discard, b_lrn_shell, masks = setup_betas(dims)
    u_lrn = 1.2 if u_lrn == 0 else u_lrn
    b_lrn = b_lrn if len(b_lrn) > 0 else b_lrn_shell
    A *= time_scale

    p_b = xp.zeros( xp.sum( dims[i]**(i+1) for i in range(len(dims)) ) )
    p_u = 0.0
    U, distances = xp.zeros(1 + epochs//ep_skip)   ,   xp.zeros((4, 1 + epochs//ep_skip))
    st_des = st_des[ xp.argwhere(xp.logical_and( B[0].size<st_des, st_des<x.size )) ].flatten()

    dx = xp.zeros(x.size)
    dx[1:] = xp.diff(x, axis=0)

    average_spike_dist = int( xp.average( xp.diff(st_des) ) )
    segment_length = B[0].size + average_spike_dist * spikes_per_epoch
    # segment_length = B[0].size * 4
    # print(segment_length, average_spike_dist, B.shape, B[0].size, u_lrn)

    zinga = []

    for epoch in range(epochs+1):
        xp.random.seed(epoch)
        st_lrn, db_th, du_th, des = simulate(x, dx, segment_length, B, b_lrn, dims, u_lrn, T, A, st_des, True)
        p_b, p_u, db, du = update(p_lr, caps, p_b, p_u, db_th, du_th)
        learn(epoch, dims, b_lrn, db, masks)
        u_lrn += du

    return u_lrn, distances, b_lrn

@get_decorator()
def test_sim_all(epochs, ep_skip, dims, p_lr, caps):
    xp.random.seed(0)
    B, b_des, b_lrn, masks = setup_betas(dims)
    u_des = 1.2
    T = 150

    xp.random.seed(0)
    x = xp.random.uniform(-1,1,20000)
    dx = xp.zeros(x.size)
    dx[1:] = xp.diff(x, axis=0) # potentially problematic (different size from x)
    st_des, db_th, du_th = sim_1D(x, dx, B, b_des, dims, u_des, T, A, xp.empty( shape=(0), dtype=xp.float64 ) )

    return learn_segments(x, st_des, dims, [], 0, epochs, ep_skip, p_lr, caps)
    # return learn_segments(x, st_des, dims, b_lrn, 0, epochs, ep_skip, p_lr, caps)

@get_decorator()
def generate_desired(seed, sig_length, dims, u_des, T, A):
    xp.random.seed(seed)
    B, b_des, b_lrn, masks = setup_betas(dims)

    xp.random.seed(seed)
    x = xp.random.normal(0,0.1,sig_length)
    dx = xp.zeros(x.size)
    dx[1:] = xp.diff(x, axis=0) # potentially problematic (different size from x)
    st_des, db_th, du_th = sim_1D(x, dx, B, b_des, dims, u_des, T, A, xp.empty( shape=(0), dtype=xp.float64 ) )

    return x, st_des, b_des
