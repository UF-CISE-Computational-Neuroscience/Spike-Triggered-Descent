from .cupy_or_numpy import xp
from .decorators import get_jit_decorator

@get_jit_decorator()
def dEdt_njit(tD, tO, i, T):
    sum1 = 0.0
    for j in range(tO.size):
        n = tO[j] * ( ( tO[j] - tO[i] ) - ( tO[i] / T ) * ( tO[j] + tO[i] ) )
        d = ( (tO[j] + tO[i]) ** 3.0 )
        e = xp.exp( -( tO[j] + tO[i] ) / T)
        sum1 += (n / d) * e

    sum2 = 0.0
    for j in range(tD.size):
        n = tD[j] * ( ( tD[j] - tO[i] ) - ( tO[i] / T ) * ( tD[j] + tO[i] ) )
        d = ( (tD[j] + tO[i]) ** 3.0 )
        e = xp.exp( -( tD[j] + tO[i] ) / T)
        sum2 += (n / d) * e
    return 2 * (sum1 - sum2)

@get_jit_decorator()
def DE_dEdt_njit(tD, tO, T):
    r = xp.zeros(tO.size)
    for i in xp.arange(tO.size):
        r[i] = -dEdt_njit(tD, tO, i, T)
    return r
