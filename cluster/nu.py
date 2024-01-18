import numpy as np
from numba import njit, prange, set_parallel_chunksize
from scipy.optimize import curve_fit

kbT = 0.4

D = 6
H = 8 * D
L = 50 * D
n = 10

v_max = 0.2

N = n * L * H
r = np.zeros((N, 2), dtype=np.float32)
v = np.random.normal(0, np.sqrt(kbT), (N, 2)).astype(np.float32)

delta = 0.25

for i in range(L):
    for j in range(H):
        r[(i*H+j)*n:(i*H+j+1)*n, :] = np.random.rand(n, 2) + np.array([i, j])

r[:, 1] = r[:, 1] - delta

intake = r[:, 0] <= 10
v[intake, 0] += 4*v_max * ((H-delta) - r[intake, 1]) * (r[intake, 1] + delta) / H**2

R_plus = np.array([[0., -1.], [1., 0.]], dtype=np.float32)
R_minus = np.array([[0., 1.], [-1., 0.]], dtype=np.float32)


@njit(parallel=True, fastmath=True)
def update(r, v):
    
    r += v

    # bounce-back rule for walls
    to_bounce = r[:, 1] <= -delta
    tac = (r[to_bounce, 1] + delta) / v[to_bounce, 1]
    r[to_bounce, 0] = r[to_bounce, 0] - 2*v[to_bounce, 0]*tac
    r[to_bounce, 1] = r[to_bounce, 1] - 2*v[to_bounce, 1]*tac
    tau = 2*np.sum(v[to_bounce, 0]) / L
    v[to_bounce] = -v[to_bounce]
    
    to_bounce = r[:, 1] > (H-delta)
    tac = (r[to_bounce, 1] - (H-delta)) / v[to_bounce, 1]
    r[to_bounce, 0] = r[to_bounce, 0] - 2*v[to_bounce, 0]*tac
    r[to_bounce, 1] = r[to_bounce, 1] - 2*v[to_bounce, 1]*tac
    tau += 2*np.sum(v[to_bounce, 0]) / L
    v[to_bounce] = -v[to_bounce]

    # pbc on x
    r[r[:, 0] > L, 0] = r[r[:, 0] > L, 0] - L
    r[r[:, 0] <= 0, 0] = r[r[:, 0] <= 0, 0] + L

    eps = np.random.rand(L*H)
    for i in prange(L):
        # first shifted cell
        cell = (r[:, 0] > i) & (r[:, 0] <= (i+1)) & (r[:, 1] > -delta) & (r[:, 1] <= 0)
        v_cell  = v[cell, :]

        u = np.sum(v_cell, axis=0) / v_cell.shape[0]

        if eps[i*H] > 0.5:
            v[cell, :] = u + np.dot(v_cell-u, R_plus)  # should transpose but symmetry
        else:
            v[cell, :] = u + np.dot(v_cell-u, R_minus) # should transpose but symmetry

        # bulk cells
        for j in range(H-1):
            cell = (r[:, 0] > i) & (r[:, 0] <= (i+1)) & (r[:, 1] > j) & (r[:, 1] <= (j+1))
            v_cell  = v[cell, :]

            u = np.sum(v_cell, axis=0) / v_cell.shape[0]

            if eps[i*H+j+1] > 0.5:
                v[cell, :] = u + np.dot(v_cell-u, R_plus)  # should transpose but symmetry
            else:
                v[cell, :] = u + np.dot(v_cell-u, R_minus) # should transpose but symmetry

        # last shifted cell
        cell = (r[:, 0] > i) & (r[:, 0] <= (i+1)) & (r[:, 1] > (H-1)) & (r[:, 1] <= (H-delta))
        v_cell  = v[cell, :]

        u = np.sum(v_cell, axis=0) / v_cell.shape[0]

        if eps[(i+1)*H-1] > 0.5:
            v[cell, :] = u + np.dot(v_cell-u, R_plus)  # should transpose but symmetry
        else:
            v[cell, :] = u + np.dot(v_cell-u, R_minus) # should transpose but symmetry
    
    intake = r[:, 0] <= 10
    v[intake, 0] = np.random.normal(0, np.sqrt(kbT), np.sum(intake))
    v[intake, 0] += 4*v_max * ((H-delta) - r[intake, 1]) * (r[intake, 1] + delta) / H**2
    return r, v, tau

@njit(fastmath=True, parallel=True)
def compute_ux(r, v):
    ux = np.zeros(H)
    for j in prange(H):
        column = (r[:, 1] > (j-delta)) & (r[:, 1] <= (j+1-delta))
        ux[j] = np.mean(v[column, 0])
    return ux


M_eq = 20000
M = 80000

columns = np.arange(H) + 0.5 - delta
ux = np.zeros((H, M-M_eq))
T = np.zeros(M-M_eq)

for i in range(M+1):
    r, v, tau = update(r, v)

    if i > M_eq:
        ux[:, i-1-M_eq] = compute_ux(r, v)
        T[i-1-M_eq] = tau

d_ux = np.std(ux, axis=1) / np.sqrt((M-M_eq)/5)
ux = np.mean(ux, axis=1)

G = np.mean(T) / H
d_G = np.std(T) / np.sqrt(M-M_eq) / H

print(f"G : {G:.6f} {d_G:.6f}")

f_fit = lambda y, A:  A / 2 * (H-delta-y)*(y+delta)

popt, pcov = curve_fit(f_fit, columns, ux, sigma=d_ux)

nus = np.zeros(2)

nu = G / popt[0] / n
d_nu = np.sqrt((d_G/G)**2 + pcov[0][0]/popt[0]**2) / n

print(f"Fit : {nu:.4f} {d_nu:.4f}")

nus[0] = nu
nus[1] = d_nu

np.save("data/nu.npy", nus)
np.save("data/ux.npy", np.vstack((ux, d_ux)).T)
