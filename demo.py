import numpy as np
import matplotlib.pyplot as plt

from split1d_cuda.cupy_or_numpy import xp
from split1d_cuda.mock_real import generate_desired, learn_segments, dist
from split1d_cuda.basis_np import Basis as basis_np
from split1d_cuda.make_k import make_k9
from split1d_cuda.constants import smoothness
from split1d_cuda.conv import nbconv_1d2o as conv
from split1d_cuda.sim import sim_1D

from STA import get_sta, scale_k

def std_kernel(cycles, x, st_des, dims, u_des, T, A, B, b_des, Kd):
    b_lrn, u_lrn = [], u_des
    epochs, ep_skip, spikes_per_epoch, time_scale = 500, 500, 5, 1
    p_lr, caps = xp.asarray([0.95, 0.92, 1, 0.0]), xp.asarray([0.01, 0.01])
    U, distances, b_lrn = learn_segments(x, st_des, dims, b_lrn, u_lrn, T, A, epochs, ep_skip, spikes_per_epoch, time_scale, p_lr, caps)
    k_epochs, b_epochs, Kl_arr = [], [], [0]

    for i in range(cycles):
        U, distances, b_lrn = learn_segments(x, st_des, dims, b_lrn, u_lrn, T, A
            , epochs, ep_skip, spikes_per_epoch, time_scale, p_lr, caps)
        u_lrn = U
        Kl = make_k9(B,b_lrn)

        bd, bl = b_des[1]/np.max(b_des[1]), b_lrn[1]/np.max(b_lrn[1])
        kdist, bdist = dist(Kd[1],Kl[1]), dist(bd,bl),
        k_epochs += [kdist]
        b_epochs += [bdist]
        Kl_arr[0] = Kl[1]

        # if 2nd and/or third order are used, ex: dims=[10,5,5]
        kdist2 = dist(Kd[2],Kl[2]) if dims[1] > 0 else 0
        kdist3 = dist(Kd[3],Kl[3]) if dims[2] > 0 else 0
        print(i, kdist, bdist, kdist2, kdist3)

    return Kl[1], {'k_epochs': k_epochs, 'b_epochs': b_epochs, 'Kl_best' : Kl_arr}


def demo():
    seed, sig_length = 0, 10000
    batches, dims, u_des, T, A = 10, xp.asarray( [20,0,0] ), 1.2, 150, 1000
    # batches, dims, u_des, T, A = 200, xp.asarray( [20,5,5] ), 1.2, 150, 1000
    x, st_des, b_des = generate_desired( seed, sig_length, dims, u_des, T, A )
    (k, B, b) = basis_np( dims[0], smoothness )
    Kd  = make_k9( B, b_des )
    u_lrn = u_des

    # perform STD on (x, st_des) with a kernel shape of dims
    kstd, learning = std_kernel( batches, x, st_des, dims, u_des, T, A, B, b_des, Kd )

    b_sta = [i.copy()*0 for i in b_des]
    ksta = get_sta(x, Kd[1].size, st_des, b_sta, B, 4)
    
    plt.title('CSRM comparison of STD and STA')
    plt.plot(scale_k(Kd), lw=5, label='des' )
    plt.plot(scale_k([0, kstd]), label='STD')
    plt.plot(ksta, label='STA')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    demo()