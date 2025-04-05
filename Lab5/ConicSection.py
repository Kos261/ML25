import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x = np.linspace(-1.5, 1.5, 100)
y = np.linspace(-1.5, 1.5, 100)
x, y = np.meshgrid(x, y)


z_conic = x**2 + y**2
z_plane = np.full_like(z_conic, 2)  # z = 1.5

theta = np.linspace(0, 2 * np.pi, 400)
r = np.sqrt(1.5)  # radius of circle at z = 1.5
x_circle = r * np.cos(theta)
y_circle = r * np.sin(theta)
z_circle = np.full_like(x_circle, 1.5)


fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z_conic, alpha=0.5, color='blue', edgecolor='none')
ax.plot_surface(x, y, z_plane, alpha=0.5, color='red', edgecolor='none')
ax.plot(x_circle, y_circle, z_circle, color='black', linewidth=2, label='Intersection')

# Labels and title
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_zlabel('$z$')
ax.set_title('Conic Section')
ax.legend()
ax.set_box_aspect([1, 1, 1.2])

plt.show()
