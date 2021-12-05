from .cupy_or_numpy import xp
# import numpy as xp
# import matplotlib.pyplot as plt

#degree = 2, derived independently to confirm 5.4
# https://www-m16.ma.tum.de/foswiki/pub/M16/Allgemeines/AdvFE16/bsplines_Ahmed.pdf
#derived again to match m=3
# https://www.sciencedirect.com/science/article/pii/S0893965910002284
#using notations from equations on page 90
# https://www.bibsonomy.org/bibtex/24d315364749a5ec96c80e03e04c313e5/jwbowers
def b(t, low):
    x = t - low
    if 0 + low <= t and t < 1 + low:
        return 0.5 * x**2
    if 1 + low <= t and t < 2 + low:
        # return 0.5 - (x-2) - (x-2)**2
        return -x**2 + 3*x - 1.5
    if 2 + low <= t and t < 3 + low:
        # return 0.5 - (x-2) + 0.5*(x-2)**2
        return 0.5 * x**2 - 3*x + 4.5
    else:
        return 0

def B(i, steps, n):
    Basis = xp.zeros(n*steps + 1)
    for t in range(len(Basis)):
        Basis[t] = b(t/(1.0*steps), i)
    return Basis/xp.sum(Basis)


def Basis(n, s):
    beta_i = xp.random.uniform(-1,1,n)
    Bi = xp.asarray([ B(i, s, n+2) for i in range(n) ])
    k = xp.einsum("ij,i->j", Bi, beta_i)
    return (k, Bi, beta_i)
