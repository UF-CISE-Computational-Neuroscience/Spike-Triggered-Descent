from .cupy_or_numpy import xp
from .decorators import get_jit_decorator, get_decorator, reset_globals, profile_results
from .constants import smoothness

@get_decorator()
def genK_nD_rs(B,b): #rectangular solid
    n = len(B)
    s = [chr(i+97) for i in range(0,2*n,2)]
    f = [chr(i+98) for i in range(0,2*n,2)]
    e = ",".join([i[0]+i[1] for i in list(zip(s,f))])
    e += ","+"".join(s)+"->"+"".join(f)
    return xp.einsum(e, *B, b.reshape([i.shape[0] for i in B]))

@get_jit_decorator()
def genK_o1_naive_njit(B0, b):
    K = xp.zeros(B0[0].size)
    for i in range(K.shape[0]):
        for u in range(len(B0)):
            K[i] += B0[u][i] * b[u]
    return K

@get_jit_decorator()
def genK_o2_naive_njit(B0, B1, b):
    K = xp.zeros((B0[0].size, B1[0].size))
    for i in range(K.shape[0]):
        for j in range(K.shape[1]):
            for u in range(len(B0)):
                for v in range(len(B1)):
                    K[i,j] += B0[u][i] * B1[v][j] * b[u,v]
    return K

@get_jit_decorator()
def genK_o3_naive_njit(B0, B1, B2, b):
    K = xp.zeros((B0[0].size, B1[0].size, B2[0].size))
    for i in range(K.shape[0]):
        for j in range(K.shape[1]):
            for k in range(K.shape[2]):
                for u in range(b.shape[0]):
                    for v in range(b.shape[1]):
                        for w in range(b.shape[2]):
                            K[i,j,k] += B0[u][i] * B1[v][j] * B2[w][k] * b[u,v,w]
    return K

@get_jit_decorator()
def genK_o3_naive_njit_v2(B0, B1, B2, b):
    K = xp.zeros((B0[0].size, B1[0].size, B2[0].size))
    offset = 11 #smoothness = 4
    for i in range(K.shape[0]):
        for j in range(min(i+offset, K.shape[1])):
            for k in range(min(j+offset, K.shape[2])):
                for u in range(b.shape[0]):
                    for v in range(b.shape[1]):
                        for w in range(b.shape[2]):
                            K[i,j,k] += B0[u][i] * B1[v][j] * B2[w][k] * b[u,v,w]
    return K

@get_jit_decorator()
def genK_o3_naive_njit_v3(B0, B1, B2, b):
    K = xp.zeros((B0[0].size, B1[0].size, B2[0].size))
    offset = 11 #smoothness = 4
    for i in range(K.shape[0]):
        for j in range(min(i+offset, K.shape[1])):
            for k in range(min(j+offset, K.shape[2])):
                for u in range(b.shape[0]):
                    for v in range(b.shape[1]):
                        for w in range(b.shape[2]):
                            K[i,j,k] += B0[u][i] * B1[v][j] * B2[w][k] * b[0]
    return K

@get_decorator()
def make_k1(B, b):
    K = [b[0]]
    K += [ genK_nD_rs([B]*i, b[i]) for i in range(1,len(b)) ]
    return K

@get_decorator()
def make_k2(B, b):
    K = [b[0]]
    K += [ genK_nD_rs([B], b[1]) ]
    s = b[1].size
    if 2 < len(b):
        K += [genK_o2_naive_njit(B, B, b[2].reshape(s,s))]
    if 3 < len(b):
        K += [genK_o3_naive_njit(B, B, B, b[3].reshape(s,s,s))]
    return K


@get_decorator()
def make_k3(B, b):
    K = [b[0]]
    K += [genK_o1_naive_njit(B, b[1])]
    s = b[1].size
    if 2 < len(b):
        K += [genK_o2_naive_njit(B, B, b[2].reshape(s,s))]
    if 3 < len(b):
        # K += [genK_o3_naive_njit_v2(B, B, B, b[3].reshape(s,s,s))]
        K += [genK_o3_naive_njit_v2(B, B, B, b[3].reshape(s,s,s))]
    return K

@get_jit_decorator()
def make_k1_unit(B):
    return B[0][0<B[0]]

@get_jit_decorator()
def make_k2_unit(B1):
    K = xp.zeros((B1.size, B1.size))
    for i in range(K.shape[0]):
        for j in range(K.shape[1]):
            K[i,j] = B1[i] * B1[j]
    return K

@get_jit_decorator()
def make_k3_unit(B1):
    K = xp.zeros((B1.size, B1.size, B1.size))
    for i in range(K.shape[0]):
        for j in range(K.shape[1]):
            for k in range(K.shape[1]):
                K[i,j,k] = B1[i] * B1[j] * B1[k]
    return K

@get_jit_decorator()
def make_k1_from_units(unit, b, smoothness):
    K = xp.zeros( (b.size+2)*smoothness+1 )
    for i in range(b.size):
        K[i*smoothness+1:i*smoothness+unit.size+1] += b[i]*unit
    return K

@get_jit_decorator()
def make_k2_from_units(unit, b, smoothness):
    K = xp.zeros( ( (b.shape[0]+2)*smoothness+1, (b.shape[0]+2)*smoothness+1 ) )
    for i in range(b.shape[0]):
        for j in range(i+1):
            K[i*smoothness+1:i*smoothness+unit.shape[0]+1
            , j*smoothness+1:j*smoothness+unit.shape[0]+1 ] += unit*b[i,j]
    return K

@get_jit_decorator()
def make_k3_from_units(unit, b, smoothness):
    s = (b.shape[0]+2)*smoothness+1
    K = xp.zeros( ( s, s, s ) )
    for i in range(b.shape[0]):
        for j in range(i+1):
            for k in range(j+1):
                K[i*smoothness+1:i*smoothness+unit.shape[0]+1
                , j*smoothness+1:j*smoothness+unit.shape[0]+1
                , k*smoothness+1:k*smoothness+unit.shape[0]+1
                ] += unit*b[i,j,k]
    return K

@get_decorator()
def make_k4(B, b, smoothness):
    k1_unit = make_k1_unit(B)
    k1 = make_k1_from_units(k1_unit, b[1], smoothness)
    k2_unit = make_k2_unit(k1_unit)
    k2 = make_k2_from_units(k2_unit, b[2].reshape(b[1].size,b[1].size), smoothness)
    k3_unit = make_k3_unit(k1_unit)
    k3 = make_k3_from_units(k3_unit, b[3].reshape(b[1].size,b[1].size,b[1].size), smoothness)
    return [b[0], k1, k2, k3]

@get_jit_decorator()
def make_k5(B, b0, b1, b2, b3, smoothness):
    k1_unit = make_k1_unit(B)
    k1 = make_k1_from_units(k1_unit, b1, smoothness)
    k2_unit = make_k2_unit(k1_unit)
    k2 = make_k2_from_units(k2_unit, b2.reshape(b1.size,b1.size), smoothness)
    k3_unit = make_k3_unit(k1_unit)
    k3 = make_k3_from_units(k3_unit, b3.reshape(b1.size,b1.size,b1.size), smoothness)
    return b0, k1, k2, k3

@get_decorator()
def make_k6(B, b, smoothness):
    k1_unit = make_k1_unit(B)
    k1 = make_k1_from_units(k1_unit, b[1], smoothness)
    k2_unit = make_k2_unit(k1_unit)
    s2 = round(b[2].size**(1/2))
    k2 = make_k2_from_units(k2_unit, b[2].reshape(s2,s2), smoothness)
    k3_unit = make_k3_unit(k1_unit)
    s3 = round(b[3].size**(1/3))
    k3 = make_k3_from_units(k3_unit, b[3].reshape(s3,s3,s3), smoothness)
    return [b[0], k1, k2, k3]

@get_jit_decorator()
def get_kl(b, r):
    return round(b.size**(1/r))

@get_decorator()
def make_k7(B, b, smoothness):
    k1_unit = make_k1_unit(B)
    k1 = make_k1_from_units(k1_unit, b[1], smoothness)
    k2_unit = make_k2_unit(k1_unit)
    s2 = get_kl(b[2], 2)
    k2 = make_k2_from_units(k2_unit, b[2].reshape(s2,s2), smoothness)
    k3_unit = make_k3_unit(k1_unit)
    s3 = get_kl(b[3], 3)
    k3 = make_k3_from_units(k3_unit, b[3].reshape(s3,s3,s3), smoothness)
    return [b[0], k1, k2, k3]

@get_decorator()
def make_k8(B, b):
    k1_unit = make_k1_unit(B)
    k1 = make_k1_from_units(k1_unit, b[1], smoothness)
    k2_unit = make_k2_unit(k1_unit)
    s2 = get_kl(b[2], 2)
    k2 = make_k2_from_units(k2_unit, b[2].reshape(s2,s2), smoothness)
    k3_unit = make_k3_unit(k1_unit)
    s3 = get_kl(b[3], 3)
    k3 = make_k3_from_units(k3_unit, b[3].reshape(s3,s3,s3), smoothness)
    return [b[0], k1, k2, k3]

@get_decorator()
def make_k9(B, b):
    k = [b[0][0]] + [xp.asarray([])]*(len(b)-1)
    k[2] = xp.asarray([[0]])
    k[3] = xp.asarray([[[0]]])
    k1_unit = make_k1_unit(B)
    k2_unit = make_k2_unit(k1_unit)
    k3_unit = make_k3_unit(k1_unit)
    if 0 < b[1].size:
        k[1] = make_k1_from_units(k1_unit, b[1], smoothness)
    if 0 < b[2].size:
        s2 = get_kl(b[2], 2)
        k[2] = make_k2_from_units(k2_unit, b[2].reshape(s2,s2), smoothness)
    if 0 < b[3].size:
        s3 = get_kl(b[3], 3)
        k[3] = make_k3_from_units(k3_unit, b[3].reshape(s3,s3,s3), smoothness)
    return k

# @get_decorator()
@get_jit_decorator()
def make_k10(B, b): #attempt
    # k = [b[0]] + [xp.asarray([])]*(len(b)-1)
    k = [b[0]] #+ [xp.zeros(1, dtype=xp.float64)]*(len(b)-1)
    k1_unit = make_k1_unit(B)
    k2_unit = make_k2_unit(k1_unit)
    k3_unit = make_k3_unit(k1_unit)
    if 0 < b[1].size:
        s1 = b[1].size
        k += [make_k1_from_units(k1_unit, b[1], smoothness).reshape(1,1,s1)]
    if 0 < b[2].size:
        s2 = get_kl(b[2], 2)
        k += [make_k2_from_units(k2_unit, b[2].reshape(1,s2,s2), smoothness)]
    if 0 < b[3].size:
        s3 = get_kl(b[3], 3)
        k = [make_k3_from_units(k3_unit, b[3].reshape(s3,s3,s3), smoothness)]
    return k

# @get_decorator()
@get_jit_decorator()
def make_k11(B, b0, b1, b2, b3):
    # k = [b[0][0]] + [xp.asarray([])]*(len(b)-1)
    k0 = xp.asarray(b0[0])
    k1 = xp.asarray([0.0])
    k2 = xp.asarray([[0.0]])
    k3 = xp.asarray([[[0.0]]])
    k1_unit = make_k1_unit(B)
    k2_unit = make_k2_unit(k1_unit)
    k3_unit = make_k3_unit(k1_unit)
    if 0 < b1.size:
        k1 = make_k1_from_units(k1_unit, b1, smoothness)
    if 0 < b2.size:
        s2 = get_kl(b2, 2)
        k2 = make_k2_from_units(k2_unit, b2.reshape(s2,s2), smoothness)
    if 0 < b3.size:
        s3 = get_kl(b3, 3)
        k3 = make_k3_from_units(k3_unit, b3.reshape(s3,s3,s3), smoothness)
    return k0, k1, k2, k3
