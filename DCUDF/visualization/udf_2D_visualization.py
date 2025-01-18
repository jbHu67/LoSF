import numpy as np
import matplotlib.pyplot as plt
import argparse


def plot_iso_surface(
    udf, axis=2, iso_surface=[0, 32, 64, 128], resolution=256, save=False
):
    """Plot the iso surface of (N, N, N) udf grid
    udf: (N, N, N)
    iso_surface: location of the intersecting surface
    return: contour plot
    """
    X, Y = np.mgrid[0:256:256j, 0:256:256j]

    _, axs = plt.subplots(2, 2)
    for i, ax in zip(range(2), axs):
        for j, ax_temp in zip(range(2), ax):
            if axis == 0:
                Z = udf[iso_surface[i * 2 + j], :, :]
            elif axis == 1:
                Z = udf[:, iso_surface[i * 2 + j], :]
            elif axis == 2:
                Z = udf[:, :, iso_surface[i * 2 + j]]
            cs = ax_temp.contourf(X, Y, Z, levels=20)
            ax_temp.contour(cs, colors="k")
            ax_temp.clabel(cs, fmt='%2.3f', colors='w', fontsize=14)
            cbar = plt.colorbar(cs)
            ax_temp.set_title("iso_surface=" + str(iso_surface[i * 2 + j]))

            # Plot grid.
            ax_temp.grid(c="k", ls="-", alpha=0.3)

    if save == True:
        plt.savefig("default.png")
    else:
        plt.show()

    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--udf_path", required=True, help="The path to the UDF numpy file"
    )
    parser.add_argument(
        "--axis", required=False, default=0, help="The axis for 2D plot"
    )

    arg = parser.parse_args()

    UDF_np = np.load(arg.udf_path)
    plot_iso_surface(UDF_np, int(arg.axis))
