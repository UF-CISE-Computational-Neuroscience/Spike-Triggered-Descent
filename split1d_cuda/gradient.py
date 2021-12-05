from .cupy_or_numpy import xp
from .decorators import get_jit_decorator, get_decorator
from .dEdt import DE_dEdt_njit, dEdt_njit
# from .E_st_relax import st_slack_nj

@get_jit_decorator()
def get_db(dEdb):
    return xp.sum(dEdb, axis=1)

@get_jit_decorator()
def DEDB_4(Dtdb, dE):
    c, bsize, b = dE.size, Dtdb.shape[0], Dtdb
    dEdb = xp.zeros((bsize, c))
    dEdb[:bsize, :c] = xp.sum( b[:bsize, :c, :c] * dE[:c], axis=2)
    return dEdb

@get_jit_decorator()
def DU(Dtdu, dE):
    C, du = dE.size, 0
    for l in range(1, C):
        for k in range(l+1, C+1):
            du -= dE[k-2] * Dtdu[(l, k)]
    return du

# Adding in relaxation as a slack parameter on the desired spike train.
@get_jit_decorator()
def gradient(tO, tDes, t, T, Dtdb, Dtdu):
    db, du, n = xp.zeros(Dtdb.shape[0]), 0, 0
    if len(tO) > n and len(tDes) > n:
        tD = tDes.copy()
        # tD = st_slack_nj(tDes, tO, 0.5)
        tO = t - tO
        tD = t - tD
        dE = DE_dEdt_njit(tD, tO, T)
        dEdb = DEDB_4(Dtdb, dE)
        db += get_db(dEdb)
        ddu = DU(Dtdu, dE)
        du += ddu
    return db, du
