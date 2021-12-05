import time
from numba import njit, jit, objmode
from .cupy_or_numpy import xp

# This is a script for profiling
# more comments on it here
# https://numba.discourse.group/t/profiling-with-a-decorator-and-njit/55


USE_TIMER = False
CACHE = True
# CACHE = False
results = {}
tree = {'stack':['main'], 'main':set()}

def wrapper_objm_start(f):
    start = time.time()
    tree[ tree['stack'][-1] ].add( f.__name__ )
    tree['stack'] += [ f.__name__ ]
    if f.__name__ not in results:
        tree[f.__name__] = set()
        # print(tree['stack'])
    return start

def wrapper_objm_end(f, start):
    run_time = time.time() - start
    if f.__name__ in results:
        results[f.__name__] += [run_time]
    else:
        results[f.__name__] = [run_time]
    tree['stack'] = tree['stack'][:-1]

def timer(f):
    def wrapper(*args, **kwargs):
        start = wrapper_objm_start(f)
        g = f(*args)
        wrapper_objm_end(f, start)
        return g
    return wrapper

def timer_none(f):
    def wrapper(*args):
        return f(*args)
    return wrapper

def jit_timer(f):
    jf = njit(f)
    @njit(cache=CACHE)
    def wrapper(*args):
        with objmode(start='float64'):
            start = wrapper_objm_start(f)
        g = jf(*args)
        # g = f(*args)
        with objmode():
            wrapper_objm_end(f, start)
        return g
    return wrapper

def get_jit_decorator():
    if USE_TIMER:
        # return timer
        return jit_timer
    else:
        return njit(cache=CACHE)

def get_decorator():
    if USE_TIMER:
        return timer
    else:
        return timer_none

def reset_globals():
    global results
    results = {}
    global tree
    tree = {'stack':['main'], 'main':set()}

def print_tree(node, layer):
    for n in node:
        rt = xp.sum(results[n])
        rtr = rt / xp.sum(results['main'])
        print('{0:>9.6f} {1:.03f}'.format( rt, rtr ), '-|-'*layer, n)
        print_tree(tree[n], layer+1)

def profile_results():
    # print(results)
    # print(tree)
    l = []
    for k in results:
        a = xp.asarray(results[k])
        # l += [[k+' '*(17-len(k)), xp.sum(a[1:])]]
        l += [[k+' '*(17-len(k)), xp.sum(a)]]
    l = sorted(l, key=lambda x: x[1])
    # for i in range(len(l)):
    #     print(  '{:.6f}'.format( l[i][1] ), l[i][0] )
        # print( l[i][0], "{:.6f}".format( l[i][1] ) )
    print_tree(tree['main'], 0)


# EXAMPLE:
# 0.003904  test_sim
# 0.003134 -|- sim_1D
# 0.001564 -|--|- gcsrm_1D
# 0.000029 -|--|--|- nbconv_1d1o
# 0.000047 -|--|--|- conv_1d1o
# 0.001256 -|--|--|- nbconv_1d2o
# 0.000079 -|--|--|- get_spikes_v2
# 0.000003 -|--|--|--|- line_intercept
# 0.000280 -|--|- partials
# 0.000154 -|--|--|- compute_dtdbd
# 0.000068 -|--|--|--|- td
# 0.000005 -|--|--|--|--|- outer
# 0.000002 -|--|--|--|--|- intk
# 0.000061 -|--|--|--|- intk_all
# 0.000002 -|--|--|--|--|- intk
# 0.000004 -|--|--|- dndu
# 0.000005 -|--|--|- dndt
# 0.000055 -|--|--|- get_d
# 0.000002 -|--|--|--|- intk
# 0.000006 -|--|--|--|- int2d1dk
# 0.000032 -|--|- genK_nD_rs