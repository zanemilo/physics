import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons
from matplotlib.colors import Normalize

# pick some reasonable min/max for |E|
norm = Normalize(vmin=0, vmax=5e6)  
# → forces your colorbar & arrows to map E_mag=0→blue, E_mag=5e6→yellow


# Coulomb constant
k = 8.988e9  # N·m²/C²

# Initial parameters
init_q1 = 2e-3
init_q2 = 5e-3
init_dist = 2.5
init_grid = 40

# Figure setup
fig, ax = plt.subplots(figsize=(6, 6))
plt.subplots_adjust(left=0.1, bottom=0.4)

# Compute field
def compute_field(q1, q2, dist, grid_points):
    charges = np.array([q1, q2])
    positions = np.array([[-dist/2, 0.0], [dist/2, 0.0]])

    x = np.linspace(-5, 5, grid_points)
    y = np.linspace(-5, 5, grid_points)
    X, Y = np.meshgrid(x, y)

    Ex = np.zeros_like(X)
    Ey = np.zeros_like(Y)
    for q, pos in zip(charges, positions):
        dx = X - pos[0]
        dy = Y - pos[1]
        r3 = (dx**2 + dy**2)**1.5 + 1e-9
        Ex += k * q * dx / r3
        Ey += k * q * dy / r3

    E_mag = np.sqrt(Ex**2 + Ey**2)
    Ex_unit = Ex / E_mag
    Ey_unit = Ey / E_mag

    return X, Y, Ex_unit, Ey_unit, E_mag, positions

# Initial field
X, Y, Ex_u, Ey_u, E_mag, pos = compute_field(init_q1, init_q2, init_dist, init_grid)
quiver = ax.quiver(X, Y, Ex_u, Ey_u, E_mag, angles='xy', scale_units='xy', scale=2.5, cmap='viridis', norm=norm)
scatter = ax.scatter(pos[:, 0], pos[:, 1], c='r', s=100)
ax.axis('equal')
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
plt.colorbar(quiver, ax=ax, label='|E| (N/C)')

# Slider axes
ax_q1 = fig.add_axes([0.1, 0.3, 0.8, 0.03])
ax_q2 = fig.add_axes([0.1, 0.25, 0.8, 0.03])
ax_dist = fig.add_axes([0.1, 0.2, 0.8, 0.03])
ax_grid = fig.add_axes([0.1, 0.15, 0.8, 0.03])

# Sliders
slider_q1 = Slider(ax_q1, 'q₁ (C)', -5e-3, 5e-3, valinit=init_q1, valfmt='%1.1e')
slider_q2 = Slider(ax_q2, 'q₂ (C)', -5e-3, 5e-3, valinit=init_q2, valfmt='%1.1e')
slider_dist = Slider(ax_dist, 'Distance (m)', 0.5, 15.0, valinit=init_dist)
slider_grid = Slider(ax_grid, 'Grid size', 20, 80, valinit=init_grid, valstep=1, valfmt='%0.0f')

# CheckButton for streamlines
ax_toggle = fig.add_axes([0.1, 0.05, 0.2, 0.05])
toggle_button = CheckButtons(ax_toggle, ['Streamlines'], [False])

# Update plot
def update(val):
    q1 = slider_q1.val
    q2 = slider_q2.val
    dist = slider_dist.val
    grid = int(slider_grid.val)
    use_streamlines = toggle_button.get_status()[0]

    X, Y, Ex_u, Ey_u, E_mag, pos = compute_field(q1, q2, dist, grid)

    ax.clear()

    if use_streamlines:
        strm = ax.streamplot(X, Y, Ex_u, Ey_u, color=E_mag, cmap='viridis', norm=norm)
    else:
        ax.quiver(X, Y, Ex_u, Ey_u, E_mag, angles='xy', scale_units='xy', scale=2.5, cmap='viridis', norm=norm)

    ax.scatter(pos[:, 0], pos[:, 1], c='r', s=100)
    ax.axis('equal')
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"q₁={q1*1e3:.1f} mC, q₂={q2*1e3:.1f} mC, dist={dist:.2f} m, grid={grid}²")
    fig.canvas.draw_idle()

# Connect events
slider_q1.on_changed(update)
slider_q2.on_changed(update)
slider_dist.on_changed(update)
slider_grid.on_changed(update)
toggle_button.on_clicked(update)

# Initial title
ax.set_title(f"q₁={init_q1*1e3:.1f} mC, q₂={init_q2*1e3:.1f} mC, dist={init_dist:.2f} m, grid={init_grid}²")

plt.show()
