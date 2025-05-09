import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Coulomb constant
k = 8.988e9  # N·m²/C²

# Initial parameters
init_q1 = 2e-3
init_q2 = 5e-3
init_dist = 2.5
init_grid = 40

# Create figure and axis
fig, ax = plt.subplots(figsize=(6, 6))
plt.subplots_adjust(left=0.1, bottom=0.35)

# Placeholder for quiver
X = Y = Ex = Ey = E_mag = None
quiver = ax.quiver([], [], [], [], [])
scatter = ax.scatter([], [], c='r', s=100)

def compute_field(q1, q2, dist, grid_points):
    # Charges and positions
    charges = np.array([q1, q2])
    positions = np.array([[-dist/2, 0.0], [dist/2, 0.0]])
    
    # Grid
    x = np.linspace(-3, 3, grid_points)
    y = np.linspace(-3, 3, grid_points)
    X, Y = np.meshgrid(x, y)
    
    # Field
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

def update(val):
    # Read slider values
    q1 = slider_q1.val
    q2 = slider_q2.val
    dist = slider_dist.val
    grid = int(slider_grid.val)
    
    # Recompute
    X, Y, Ex_u, Ey_u, E_mag, pos = compute_field(q1, q2, dist, grid)
    quiver.set_offsets(np.c_[X.flatten(), Y.flatten()])
    quiver.set_UVC(Ex_u.flatten(), Ey_u.flatten(), E_mag.flatten())
    scatter.set_offsets(pos)
    ax.set_title(f"q₁={q1*1e3:.1f} mC, q₂={q2*1e3:.1f} mC, dist={dist:.2f} m, grid={grid}²")
    fig.canvas.draw_idle()

# Initial draw
X, Y, Ex_u, Ey_u, E_mag, pos = compute_field(init_q1, init_q2, init_dist, init_grid)
quiver = ax.quiver(X, Y, Ex_u, Ey_u, E_mag, angles='xy', scale_units='xy', scale=2.5, cmap='viridis')
scatter = ax.scatter(pos[:, 0], pos[:, 1], c='r', s=100)
ax.set_title("Electric Field Visualization for Two Point Charges")
ax.axis('equal')
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_title(f"q₁={init_q1*1e3:.1f} mC, q₂={init_q2*1e3:.1f} mC, dist={init_dist:.2f} m, grid={init_grid}²")
plt.colorbar(quiver, ax=ax, label='|E| (N/C)')

# Sliders
ax_q1 = fig.add_axes([0.1, 0.25, 0.8, 0.03])
ax_q2 = fig.add_axes([0.1, 0.20, 0.8, 0.03])
ax_dist = fig.add_axes([0.1, 0.15, 0.8, 0.03])
ax_grid = fig.add_axes([0.1, 0.10, 0.8, 0.03])

slider_q1 = Slider(ax_q1, 'q₁ (C)', -5e-3, 5e-3, valinit=init_q1, valfmt='%1.1e')
slider_q2 = Slider(ax_q2, 'q₂ (C)', -5e-3, 5e-3, valinit=init_q2, valfmt='%1.1e')
slider_dist = Slider(ax_dist, 'Distance (m)', 0.5, 15.0, valinit=init_dist)
slider_grid = Slider(ax_grid, 'Grid size', 40, 40, valinit=init_grid,
                     valstep=1, valfmt='%0.0f')


# Link sliders to update
slider_q1.on_changed(update)
slider_q2.on_changed(update)
slider_dist.on_changed(update)
slider_grid.on_changed(update)

plt.show()
