import numpy as np
from split1d_cuda.make_k import make_k7

def scale_k(k):
    return k[1] / max( k[1].max(), abs(k[1].min()) )

def pseudoinv(B, A):
    # convert to betas
    # min || [B] b - STA ||^2
    # [B]^T[B] b = [B]^TSTA
    # b = ([B]^T[B])^-1 [B]^TSTA
    BBinv = np.linalg.inv( np.matmul(np.transpose(B), B) )
    BBinvA = np.matmul( BBinv, np.matmul(np.transpose(B), A) )
    # print(BBinvA/np.max(BBinvA))
    # return np.matmul(B, BBinvA)
    return BBinvA

def sta(x, kl, st):
    # add windows before spikes
    ids = st[ np.logical_and( st > kl, st < x.size) ].astype(int)
    a = np.zeros(kl)
    for i in ids:
        a += x[i-kl:i]
    return a/ids.size

def make_tower(x, l):
    # matrix of consecutive signal windows
    xl = x.size
    X = np.zeros((xl-l+1, l))
    for i in np.arange( l, xl+1 ):
        X[i-l,:] = x[i-l:i][::-1] #[i:i-l:-1]
    return X

def sta_remake(x, l, st_des):
    X = make_tower(x,l)
    A = sta(x, l, st_des)
    XXinv = np.linalg.inv( np.matmul(np.transpose(X), X) )
    return np.matmul( XXinv, A )[::-1]

def make_ksta(bsta, B, XXinvA, smooth):
    bsta[1] = pseudoinv(np.transpose(B), XXinvA)
    ksta = make_k7(B, bsta, smooth)
    return scale_k(ksta)

def get_sta(x, k_size, st_des, bsta_shell, B, smooth):
    XXinvA = sta_remake(x, k_size, st_des)
    return make_ksta(bsta_shell, B, XXinvA, smooth)