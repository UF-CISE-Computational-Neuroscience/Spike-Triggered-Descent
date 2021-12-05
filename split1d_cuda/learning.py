from .cupy_or_numpy import xp
from .decorators import get_jit_decorator, get_decorator

# @get_decorator()
@get_jit_decorator()
def cap(d, cap):
    N = xp.linalg.norm( d )
    if cap < N:
        d = (d / N) * cap
    return d

# @get_decorator()
@get_jit_decorator()
def update(p_lr, caps, p_b, p_u, db_th, du_th):
    βb, βu, αb, αu = p_lr
    db = -db_th
    du = -du_th
    db = cap(db, caps[0])
    du = min( max(du, -caps[1]),  caps[1])
    return βb * p_b + db, βu * p_u + du, αb * p_b, αu * p_u

@get_decorator()
def mask_diagonal_iband(n):
    # for ij remove ji (these terms can simply be combined)
    return (xp.tri(n,n)-xp.tri(n,n,-n)).flatten()

@get_decorator()
def mask_diagonal_iband_3(n):
    y = xp.zeros((n,n,n))
    for i in range(n):
        for j in range(i+1):
            for k in range(j+1):
                y[i,j,k] = 1
    return y.flatten()

@get_decorator()
def make_bdim(dim):
    return xp.random.uniform(-1, 1, xp.prod(dim))

@get_decorator()
def make_b(dim, m):
    return [1] + [make_bdim(dim[:i+1])*m  for i in range(len(dim))]

@get_decorator()
def make_b2(dims, m):
    return [xp.asarray([1])] + [make_bdim( [dims[i]]*(i+1) )*m  for i in range(len(dims))]
