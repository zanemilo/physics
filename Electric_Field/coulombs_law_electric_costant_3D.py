from functools import lru_cache
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider

# Physical constant
epsilon_0    = 8.854187817e-12  # vacuum permittivity (F/m)
four_pi_eps0 = 4 * np.pi * epsilon_0

@lru_cache(maxsize=256)
def compute_field_cached(sep, q1, q2, grid, domain):
    axis = np.linspace(-domain, domain, grid)
    X, Y, Z = np.meshgrid(axis, axis, axis, indexing='ij')
    Ex = np.zeros_like(X); Ey = np.zeros_like(Y); Ez = np.zeros_like(Z)

    charges   = np.array([q1, q2])
    positions = np.array([[-sep/2, 0.0, 0.0], [sep/2, 0.0, 0.0]])
    for q, pos in zip(charges, positions):
        dx = X - pos[0]
        dy = Y - pos[1]
        dz = Z - pos[2]
        r3 = (dx**2 + dy**2 + dz**2)**1.5 + 1e-9
        Ex += (q / four_pi_eps0) * dx / r3
        Ey += (q / four_pi_eps0) * dy / r3
        Ez += (q / four_pi_eps0) * dz / r3

    E_mag = np.sqrt(Ex**2 + Ey**2 + Ez**2)
    return X, Y, Z, Ex, Ey, Ez, E_mag

# Initial values
init_sep, init_q1, init_q2 = 2.5, 2e-3, 5e-3
init_grid, init_skip       = 25, 2
init_domain                = 40  # fixed starting domain

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

def update(val):
    ax.cla()
    sep    = slider_sep.val
    q1     = slider_q1.val
    q2     = slider_q2.val
    grid   = int(slider_grid.val)
    skip   = int(slider_skip.val)
    domain = slider_dom.val

    X, Y, Z, Ex, Ey, Ez, E_mag = compute_field_cached(sep, q1, q2, grid, domain)

    # LogNorm for color mapping (5th–95th percentile)
    vmin, vmax = np.percentile(E_mag, [5, 95])
    norm = LogNorm(vmin=vmin, vmax=vmax)

    # Subsample vectors
    Xs = X[::skip, ::skip, ::skip].ravel()
    Ys = Y[::skip, ::skip, ::skip].ravel()
    Zs = Z[::skip, ::skip, ::skip].ravel()
    Exs= Ex[::skip, ::skip, ::skip].ravel()
    Eys= Ey[::skip, ::skip, ::skip].ravel()
    Ezs= Ez[::skip, ::skip, ::skip].ravel()
    Eflat = E_mag[::skip, ::skip, ::skip].ravel()

    # Color + alpha by radius
    colors = plt.cm.viridis(norm(Eflat))
    rs = np.sqrt(Xs**2 + Ys**2 + Zs**2)
    alphas = np.clip(1 - rs/domain, 0.1, 1.0)
    colors[:, 3] = alphas

    # Draw quiver
    ax.quiver(
        Xs, Ys, Zs,
        Exs, Eys, Ezs,
        length=domain*0.1,
        normalize=True,
        colors=colors
    )

    # Draw the two charges at ±sep/2
    ax.scatter(
        [-sep/2, sep/2],
        [0, 0],
        [0, 0],
        color='red',
        s=100
    )

    # Axes limits and labels
    ax.set_xlim(-domain, domain)
    ax.set_ylim(-domain, domain)
    ax.set_zlim(-domain, domain)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('3D Electric Field of Two Point Charges')

    # Only add colorbar once
    if not hasattr(update, "colorbar"):
        mappable = plt.cm.ScalarMappable(norm=norm, cmap='viridis')
        mappable.set_array(E_mag.ravel())
        update.colorbar = fig.colorbar(mappable, ax=ax, shrink=0.6, pad=0.1, label='|E| (V/m)')
    else:
        update.colorbar.mappable.set_norm(norm)
        update.colorbar.mappable.set_array(E_mag.ravel())

    plt.draw()

# Hook up sliders
for s in (slider_sep, slider_q1, slider_q2, slider_grid, slider_skip, slider_dom):
    s.on_changed(update)

# Initial draw
update(None)
plt.show()
