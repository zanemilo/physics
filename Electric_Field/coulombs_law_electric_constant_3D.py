import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.widgets import Slider
from numba import njit, prange

# Physical constant
epsilon_0    = np.float32(8.854187817e-12)
four_pi_eps0 = np.float32(4 * np.pi) * epsilon_0

@njit(parallel=True, fastmath=True)
def field_two_point_charges(x, y, z, sep, q1, q2, four_pi_eps0):
    n = x.size
    Ex = np.zeros(n, np.float32)
    Ey = np.zeros(n, np.float32)
    Ez = np.zeros(n, np.float32)

    px1 = -sep/2.0
    px2 =  sep/2.0

    for i in prange(n):
        dx1 = x[i] - px1; dy1 = y[i]; dz1 = z[i]
        r31 = (dx1*dx1 + dy1*dy1 + dz1*dz1)**1.5 + 1e-9
        s1  = (q1 / four_pi_eps0) / r31
        Ex[i] += s1 * dx1; Ey[i] += s1 * dy1; Ez[i] += s1 * dz1

        dx2 = x[i] - px2; dy2 = y[i]; dz2 = z[i]
        r32 = (dx2*dx2 + dy2*dy2 + dz2*dz2)**1.5 + 1e-9
        s2  = (q2 / four_pi_eps0) / r32
        Ex[i] += s2 * dx2; Ey[i] += s2 * dy2; Ez[i] += s2 * dz2

    return Ex, Ey, Ez

def make_sampled_axis(domain, grid, skip):
    n = int(np.ceil(grid/skip))
    n = max(n, 3)
    axis = np.linspace(-domain, domain, n, dtype=np.float32)
    return axis

# --- Initial values ---
init_sep, init_q1, init_q2 = 2.5, 2e-3, 5e-3
init_grid, init_skip       = 25, 2
init_domain                = 40  # starting domain

fig = plt.figure(figsize=(8, 8))
ax  = fig.add_subplot(111, projection='3d')

# Slider axes
ax_sep   = fig.add_axes([0.15, 0.02, 0.65, 0.02])
ax_q1    = fig.add_axes([0.15, 0.06, 0.65, 0.02])
ax_q2    = fig.add_axes([0.15, 0.10, 0.65, 0.02])
ax_grid  = fig.add_axes([0.15, 0.14, 0.65, 0.02])
ax_skip  = fig.add_axes([0.15, 0.18, 0.65, 0.02])
ax_dom   = fig.add_axes([0.15, 0.22, 0.65, 0.02])

slider_sep   = Slider(ax_sep,  'Separation (m)', 0.5, 100.0, valinit=init_sep)
slider_q1    = Slider(ax_q1,   'Charge 1 (C)', -5e-3, 5e-3, valinit=init_q1, valfmt='%1.1e')
slider_q2    = Slider(ax_q2,   'Charge 2 (C)', -5e-3, 5e-3, valinit=init_q2, valfmt='%1.1e')
slider_grid  = Slider(ax_grid, 'Grid Size',       10,    50, valinit=init_grid, valstep=5)
slider_skip  = Slider(ax_skip, 'Vector Skip',     1,     5,  valinit=init_skip, valstep=1)
slider_dom   = Slider(ax_dom,  'Domain (m)',      5.0,  100.0, valinit=init_domain)

cmap = plt.cm.get_cmap('viridis')
_last_draw = 0.0

def update(val):
    global _last_draw
    import time
    now = time.time()
    if now - _last_draw < 0.03:  # debounce ~30 ms
        return
    _last_draw = now

    ax.cla()
    sep    = np.float32(slider_sep.val)
    q1     = np.float32(slider_q1.val)
    q2     = np.float32(slider_q2.val)
    grid   = int(slider_grid.val)
    skip   = int(slider_skip.val)
    domain = float(slider_dom.val)

    # Sampled grid directly
    axis = make_sampled_axis(domain, grid, skip)
    X, Y, Z = np.meshgrid(axis, axis, axis, indexing='ij')
    x = X.ravel(); y = Y.ravel(); z = Z.ravel()

    Ex, Ey, Ez = field_two_point_charges(x, y, z, float(sep), float(q1), float(q2), float(four_pi_eps0))
    E_mag = np.sqrt(Ex*Ex + Ey*Ey + Ez*Ez)

    vmin, vmax = np.percentile(E_mag, [5, 95])
    norm = LogNorm(vmin=max(vmin, 1e-12), vmax=max(vmax, vmin*10))

    colors = cmap(norm(E_mag))
    rsq = x*x + y*y + z*z
    alpha = 1.0 - np.minimum(rsq, domain*domain)/(domain*domain)
    alpha = np.clip(alpha, 0.1, 1.0)
    colors[:, 3] = alpha

    ax.quiver(x, y, z, Ex, Ey, Ez, length=domain*0.1, normalize=True, colors=colors)

    # Charges
    ax.scatter([-float(sep)/2, float(sep)/2], [0,0], [0,0], color='red', s=100)

    ax.set_xlim(-domain, domain); ax.set_ylim(-domain, domain); ax.set_zlim(-domain, domain)
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
    ax.set_title('3D Electric Field of Two Point Charges')

    if not hasattr(update, "colorbar"):
        mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        mappable.set_array(E_mag)
        update.colorbar = fig.colorbar(mappable, ax=ax, shrink=0.6, pad=0.1, label='|E| (V/m)')
    else:
        update.colorbar.mappable.set_norm(norm)
        update.colorbar.mappable.set_array(E_mag)

    plt.draw()

# Hook up sliders
for s in (slider_sep, slider_q1, slider_q2, slider_grid, slider_skip, slider_dom):
    s.on_changed(update)

update(None)
plt.show()
