from .cupy_or_numpy import xp
from .decorators import get_jit_decorator

@get_jit_decorator()
def ahp(x, t, u, A):
    return -A * xp.sum( xp.exp( (x-t) / u ) )

@get_jit_decorator()
def line_intercept(y1, y2, thresh):
    #this is really a ratio
    return 1 - (thresh-y1)/(y2-y1)

@get_jit_decorator()
def get_spikes_1(c, K0, u, A):
    st = []
    y = xp.zeros(c.size)
    for xt in range(0, c.size):
        a = ahp(xp.asarray(st), xt, u, A)
        y[xt] = c[xt] + a
        if K0 < y[xt]:
            li = line_intercept(y[xt-1], y[xt], K0)
            st.append(xt-li)
    return (xp.asarray(st), y)

@get_jit_decorator()
def get_spikes_v2_2_ahp(count, st, xt, u, A):
    a = 0
    for i in range(count):
        a += xp.exp( (st[i] - xt)/u )
    a *= -A
    return a

@get_jit_decorator()
def get_spikes_2(c, K0, u, A):
    st = xp.zeros(c[K0<c].size)
    y = xp.zeros(c.size)
    count = 0
    for xt in range(0, c.size):
        if K0 < c[xt]:
            a_after = get_spikes_v2_2_ahp(count, st, xt, u, A)
            y[xt] = c[xt] + a_after
            if K0 < y[xt]:
                a_before = get_spikes_v2_2_ahp(count, st, xt-1, u, A)
                y[xt-1] = c[xt-1] + a_before
                if y[xt-1] < K0:
                    li = 1 - (K0-y[xt-1])/(y[xt]-y[xt-1])
                    st[count] = xt-li
                    count += 1
    return st[:count], y
