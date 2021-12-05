from .cupy_or_numpy import xp
from .decorators import get_jit_decorator, get_decorator
from .intk import intk1d1o, intk1d2o, intk1d3o

@get_jit_decorator()
def dndt(x, t, u, A):
    return -A * xp.exp((x-t)/u) * (-1.0 / u)

@get_jit_decorator()
def dndu(x, t, u, A):
    return -A * xp.exp((x-t)/u) * (1 / (u**2))

@get_jit_decorator()
def intk_all(B, x, xt):
    y = xp.zeros(B.shape[0])
    for i in range(y.size):
        y[i] = intk1d1o(B[i,:],x,xt)
    return -y

@get_jit_decorator()
def outer_2(a,b):
    y = xp.zeros((a.size,b.size))
    for i in range(a.size):
        for j in range(b.size):
            y[i,j] = a[i] * b[j]
    return y.flatten()

@get_jit_decorator()
def outer_3(a,b,c):
    y = xp.zeros((a.size, b.size, c.size))
    for i in range(a.size):
        for j in range(b.size):
            for k in range(c.size):
                y[i,j,k] = a[i] * b[j] * c[k]
    return y.flatten()

@get_jit_decorator()
def td(B, x, xt):
    y = xp.zeros(B.shape[0])
    for i in range(y.size):
        y[i] = intk1d1o(B[i,:], x, xt)
    return -outer_2(y,y).flatten()

@get_jit_decorator()
def compute_dtdbd(B, x, xt, dims):
    y = -intk_all(B, x, xt)
    a = xp.zeros(1)
    if 0 < dims[0]:
        a = xp.concatenate( ( a , -y[:dims[0]] ) )
    if 0 < dims[1]:
        y2 = -outer_2(y[:dims[1]],y[:dims[1]])
        a = xp.concatenate( ( a , y2 ) )
        # a = xp.concatenate( ( a , td(B,x,xt) ) ) #same thing!
    if 0 < dims[2]:
        y3 = -outer_3(y[:dims[2]],y[:dims[2]],y[:dims[2]])
        a = xp.concatenate( ( a , y3 ) )
    return a[1:]

@get_decorator()
def get_d(K, x, dx, dn_dt, xt):
    d = xp.sum(dn_dt)
    if 0 < K[1].size:
        d += intk1d1o(K[1], dx, xt)
    if 0 < K[2].size:
        d += intk1d2o(K[2], x, dx, xt)
        d += intk1d2o(K[2], dx, x, xt)
    if 0 < K[3].size:
        # possibly 3*intk1d3o ?
        d += intk1d3o(K[3], dx, x, x, xt)
        d += intk1d3o(K[3], x, dx, x, xt)
        d += intk1d3o(K[3], x, x, dx, xt)
    return d if d!=0 else 1

@get_jit_decorator()
def get_d2(K1, K2, K3, x, dx, dn_dt, xt):
    d = xp.sum(dn_dt)
    if 1 < K1.size:
        d += intk1d1o(K1, dx, xt)
    if 1 < K2.size:
        d += intk1d2o(K2, x, dx, xt)
        d += intk1d2o(K2, dx, x, xt)
    if 1 < K3.size:
        # possibly 3*intk1d3o ?
        d += intk1d3o(K3, dx, x, x, xt)
        d += intk1d3o(K3, x, dx, x, xt)
        d += intk1d3o(K3, x, x, dx, xt)
    return d if d!=0 else 1

@get_jit_decorator()
def compute_dtdb(b, dtdbd, dtdt):
    c = dtdt.size
    s = b.shape[0]
    b[:s, c, c] = dtdbd[:s]
    b[:s, :c, c] += xp.sum(b[:s, :c, :c] * dtdt[:c], axis=2)

@get_jit_decorator()
def compute_dtdb_v2(b, dtdbd, dtdt):
    c = dtdt.size
    s = b.shape[0]
    for i in range(s):
        b[i, c, c] = dtdbd[i]
    for i in range(s):
        for j in range(c):
            b[i, j, c] += b[i, j, j] * dtdt[j]

@get_jit_decorator()
def compute_dtdu_exp2(u, dtdu, dtdt):
    c = dtdt.size
    u[:c, c-1] = xp.transpose(dtdu)
    u[:c, c] += xp.sum(u[:c, :c] * dtdt[:c], axis=1)

@get_jit_decorator()
def compute_dtdu_exp1(Dtdu, dtdu, dtdt):
    C = dtdt.size+1
    for l in xp.arange(1, C):
        Dtdu[l,C] = dtdu[l-1]
        for j in xp.arange(l+1, C):
            Dtdu[l,C] += Dtdu[l,j] * dtdt[j - 2]

# @get_decorator()
# def partials(x, dx, K, B, dims, st, u, A, Dtdb, Dtdu):
@get_jit_decorator()
def partials(x, dx, K1, K2, K3, B, dims, st, u, A, Dtdb, Dtdu):
    xt = int(st[-1]+1)
    dtdt = dndt( st[:-1], xt, u, A )
    dtdu = -dndu( st[:-1], xt, u, A )
    # d = get_d(K, x, dx, dtdt, xt)
    d = get_d2(K1, K2, K3, x, dx, dtdt, xt)

    dtdt /= d #dn/dt -> dt/dt
    dtdu /= d #dn/du -> dt/du

    dtdbd = compute_dtdbd(B, x, xt, dims )/d
    # compute_dtdb(Dtdb, dtdbd, dtdt)
    compute_dtdb_v2(Dtdb, dtdbd, dtdt)
    compute_dtdu_exp1(Dtdu, dtdu, dtdt)
    # compute_dtdu_exp2(Dtdu, dtdu, dtdt) #not tested
