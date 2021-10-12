# original R code found at: https://people.sc.fsu.edu/~jburkardt/r_src/wave/wave.R

import numpy as np

def wave( u:np.ndarray, alpha:float, xdelta:float, tdelta:float, n:int )-> np.ndarray:

    m = u.size
    uarray = u.flatten()
    newu = u

    h = (alpha * tdelta / xdelta)**2

    oldu = np.zeros(m)
    oldu[1:(m-2)] = u[1:(m-2)] + h * ( u[0:(m-3)] - 2.0 * u[1:(m-2)] + u[2:m-1] ) / 2.0

    for _ in range(n):
        ustep1 = 2.0* u - oldu
        ustep2 = u[0:(m-2)] - 2.0 * u[1:(m-1)] + u[2:m]

        ustep2 = np.concatenate([[0.0], ustep2, [0.0]])
        newu = ustep1 + h * ustep2
        oldu, u = u, newu
        uarray = np.vstack((uarray, u))

    return uarray


speed = 2.0
x0 = 0.0
xdelta = 0.05
x = np.arange(x0, 1+xdelta, xdelta)
m = x.size
u = np.sin(x * np.pi * 2.0)
u[10:21] = 0.0
tdelta = 0.02
n = 40

z = wave (u, speed, xdelta, tdelta, n)

print(z)
print(z.shape)