from .decorators import get_jit_decorator

@get_jit_decorator()
def intk1d1o(k, x, tl):
    s = 0
    for i in range(k.size):
        s += k[i] * x[tl-i]
    return s

@get_jit_decorator()
def intk1d2o(k, x1, x2, tl):
    s = 0
    for i in range(k.shape[0]):
        for j in range(k.shape[1]):
            s += k[i,j] * x1[tl-i] * x2[tl-j]
    return s

@get_jit_decorator()
def intk1d3o(k, x1, x2, x3, tl):
    s = 0
    for i in range(k.shape[0]):
        for j in range(k.shape[1]):
            for l in range(k.shape[2]):
                s += k[i,j,l] * x1[tl-i] * x2[tl-j] * x3[tl-l]
    return s
