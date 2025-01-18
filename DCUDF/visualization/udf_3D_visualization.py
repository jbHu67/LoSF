import plotly.graph_objects as go
import numpy as np
import argparse


def plot_udf_3D(udf):
    X, Y, Z = np.mgrid[0:128:128j, 0:128:128j, 0:128:128j]

    fig = go.Figure(data=go.Volume(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=udf.flatten(),
        isomin=0.1,
        isomax=0.8,
        opacity=0.1,  # needs to be small to see through all surfaces
        surface_count=17,  # needs to be a large number for good volume rendering
    ))
    fig.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--udf_path',
        required=True,
        help='The path to the UDF numpy file')
    arg = parser.parse_args()

    UDF_np = np.load(arg.udf_path)
    plot_udf_3D(UDF_np)
