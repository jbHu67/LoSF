# Import required modules
import numpy as np
import matplotlib.pyplot as plt


def plot_udf_grident_2D(
        udf_gradient,
        iso_surface=100,
        field_range=128,
        plane='XY'):

    # Meshgrid
    x, y = np.meshgrid(np.linspace(0, field_range, field_range),
                       np.linspace(0, field_range, field_range))

    # Directional vectors from UDF gradient field
    udf_gradient_x = udf_gradient[:, :, :, 0]
    udf_gradient_y = udf_gradient[:, :, :, 1]
    udf_gradient_z = udf_gradient[:, :, :, 2]

    if plane == 'XY':
        u = udf_gradient_x[iso_surface, :, :]
        v = udf_gradient_y[iso_surface, :, :]

    # Plotting Vector Field with QUIVER
    plt.quiver(x, y, u, v, color='g')
    plt.title('Vector Field')

    # Setting x, y boundary limits
    plt.xlim(-5, field_range + 5)
    plt.ylim(-5, field_range + 5)

    # Show plot with grid
    plt.grid()
    plt.show()
