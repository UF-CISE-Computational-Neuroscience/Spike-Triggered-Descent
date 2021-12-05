import numpy as np

# Unresolved stability issue with np.exp(-add/c)
def summer(a,b,c):
    mul = np.multiply.outer(a,b)
    add = np.add.outer(a,b)
    add_2 = np.power(add,2)
    div = np.divide(mul,add_2, where=add_2!=0)
    # rare overflow warning
    exp = np.exp(-add/c)
    v = np.multiply(div, exp)
    s = np.einsum("ij->", v)
    return s

def E(tD,tO,T):
    tD = np.asarray(tD, np.double)
    tO = np.asarray(tO, np.double)
    return summer(tD,tD,T) + summer(tO,tO,T) - 2 * summer(tD,tO,T)
