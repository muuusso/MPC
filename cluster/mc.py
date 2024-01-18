import time

import numpy as np
from numba import njit, prange

kbT = 0.4

D = 12
H = 8 * D
L = 50 * D
n = 10

v_max = 0.2

N = n * L * H
r = np.zeros((N, 2), dtype=np.float32)
v = np.random.normal(0, np.sqrt(kbT), (N, 2)).astype(np.float32)

y0 = (H-D)/2
x0 = L/4 - D/2

delta = 0.25

for i in range(L):
    for j in range(H):
        if (i >= x0) and (i < x0+D) and (j >= y0) and (j < y0+D):
            continue
        r[(i*H+j)*n:(i*H+j+1)*n, :] = np.random.rand(n, 2) + np.array([i, j])

r[:, 1] = r[:, 1] - delta
in_D = (r[:, 1] > y0) & (r[:, 1] < (y0+D)) & (r[:, 0] > x0) & (r[:, 0] < (x0+D))
r[in_D, 1] = r[in_D, 1] - D

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
    v[to_bounce] = -v[to_bounce]
    
    to_bounce = r[:, 1] > (H-delta)
    tac = (r[to_bounce, 1] - (H-delta)) / v[to_bounce, 1]
    r[to_bounce, 0] = r[to_bounce, 0] - 2*v[to_bounce, 0]*tac
    r[to_bounce, 1] = r[to_bounce, 1] - 2*v[to_bounce, 1]*tac
    v[to_bounce] = -v[to_bounce]

    # pbc on x
    r[r[:, 0] > L, 0] = r[r[:, 0] > L, 0] - L
    r[r[:, 0] <= 0, 0] = r[r[:, 0] <= 0, 0] + L

    # bounce-back rule for cilinder
    t_x0 = (r[:, 0] - x0) / v[:, 0]
    t_y0 = (r[:, 1] - y0) / v[:, 1]
    t_x0D = (r[:, 0] - (x0+D)) / v[:, 0]
    t_y0D = (r[:, 1] - (y0+D)) / v[:, 1]
    
    in_x0 = (t_x0 < 1) & (t_x0 > 0)
    y_in_x0 = r[:, 1] - v[:, 1] * t_x0
    in_x0 = in_x0 & (y_in_x0 < (y0+D)) & (y_in_x0 > y0)
    
    in_x0D = (t_x0D < 1) & (t_x0D > 0)
    y_in_x0D = r[:, 1] - v[:, 1] * t_x0D
    in_x0D = in_x0D & (y_in_x0D < (y0+D)) & (y_in_x0D > y0)
    
    in_y0 = (t_y0 < 1) & (t_y0 > 0)
    x_in_y0 = r[:, 0] - v[:, 0] * t_y0
    in_y0 = in_y0 & (x_in_y0 < (x0+D)) & (x_in_y0 > x0)
    
    in_y0D = (t_y0D < 1) & (t_y0D > 0)
    x_in_y0D = r[:, 0] - v[:, 0] * t_y0D
    in_y0D = in_y0D & (x_in_y0D < (x0+D)) & (x_in_y0D > x0)
    
    both = (in_x0 & in_y0)
    in_x0[both] = t_x0[both] > t_y0[both]
    in_y0[both] = t_x0[both] < t_y0[both]
    
    both = (in_x0 & in_y0D)
    in_x0[both] = t_x0[both] > t_y0D[both]
    in_y0D[both] = t_x0[both] < t_y0D[both]
    
    both = (in_x0D & in_y0D)
    in_x0D[both] = t_x0D[both] > t_y0D[both]
    in_y0D[both] = t_x0D[both] < t_y0D[both]
    
    both = (in_x0D & in_y0)
    in_x0D[both] = t_x0D[both] > t_y0[both]
    in_y0[both] = t_x0D[both] < t_y0[both]
    
    r[in_x0, 0] = r[in_x0, 0] - 2*v[in_x0, 0]*t_x0[in_x0]
    r[in_x0, 1] = r[in_x0, 1] - 2*v[in_x0, 1]*t_x0[in_x0]
    
    r[in_y0, 0] = r[in_y0, 0] - 2*v[in_y0, 0]*t_y0[in_y0]
    r[in_y0, 1] = r[in_y0, 1] - 2*v[in_y0, 1]*t_y0[in_y0]
    
    r[in_x0D, 0] = r[in_x0D, 0] - 2*v[in_x0D, 0]*t_x0D[in_x0D]
    r[in_x0D, 1] = r[in_x0D, 1] - 2*v[in_x0D, 1]*t_x0D[in_x0D]
    
    r[in_y0D, 0] = r[in_y0D, 0] - 2*v[in_y0D, 0]*t_y0D[in_y0D]
    r[in_y0D, 1] = r[in_y0D, 1] - 2*v[in_y0D, 1]*t_y0D[in_y0D]

    f_par = 2*np.sum(v[in_x0, 0]) + 2*np.sum(v[in_x0D, 0])
    f_par += 2*np.sum(v[in_y0, 0]) + 2*np.sum(v[in_y0D, 0])

    v[in_x0, :] = -v[in_x0, :]
    v[in_y0, :] = -v[in_y0, :]
    v[in_x0D, :] = -v[in_x0D, :]
    v[in_y0D, :] = -v[in_y0D, :]

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
        for j in prange(H-1):
            if (i >= x0) and (i < (x0+D)) and (j >= y0) and (j < (y0+D)):
                continue

            # x shift and y shift
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
    return r, v, f_par


n_probes = 31
@njit(fastmath=True, parallel=True)
def v_rec_zone(r, v):
    v_rec = np.zeros(n_probes)
    for j in prange(n_probes):
        in_rec = (r[:, 1] > (y0+D/2-0.1)) & (r[:, 1] < (y0+D/2+0.1)) & (r[:, 0]> (x0+D+0.1*D*j)) & (r[:, 0] < (x0+D+0.1*D*(j+1)))
        if np.sum(in_rec) > 0:
            v_rec[j] = np.mean(v[in_rec, 0])
        else:
            v_rec[j] = np.nan
    return v_rec

M = 100000

r = np.load(f"data/rD{D}_eq.npy")
v = np.load(f"data/vD{D}_eq.npy")

time.sleep(3*np.random.random())

try:
    m = np.load(f"data/f_parD{D}.npy").shape[0]
except:
    m = 0
    np.save(f"data/v_recD{D}.npy", np.zeros((0, n_probes)))
    np.save(f"data/f_parD{D}.npy", np.zeros(0))
    
save_steps = np.random.randint(8, 12)
v_rec = np.zeros((save_steps, n_probes))
f_par = np.zeros(save_steps)

i = 0
while m < M:
    r, v, f = update(r, v)
    
    v_rec[i, :] = v_rec_zone(r, v) 
    f_par[i] = f

    i += 1
    if i % save_steps == 0:
        v_rec_tot = np.load(f"data/v_recD{D}.npy")
        f_par_tot = np.load(f"data/f_parD{D}.npy")

        m = f_par_tot.shape[0] + save_steps

        np.save(f"data/v_recD{D}.npy", np.vstack([v_rec_tot, v_rec]))
        np.save(f"data/f_parD{D}.npy", np.hstack([f_par_tot, f_par]))
        
        i = 0    
        v_rec = np.zeros((save_steps, n_probes))
        v_par = np.zeros(save_steps)
