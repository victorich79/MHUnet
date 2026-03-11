import pyvista as pv
from PIL import Image
from typing import List, Literal
from matplotlib.pyplot import cm
from pyvista.core.pointset import PolyData


def generate_rotating_snapshots(
    geometry: PolyData,
    rotation_step: int,
    rotation_axis: Literal["x", "y", "z"],
    clim: List[float],
    ambient: float,
    save_path: str
):
    """
    Generate rotating snapshots of a 3D geometry.

    Parameters
    ----------
    geometry : vtk.vtkPolyData
        The input geometry to be visualized.
    rotation_step : int
        The rotation step in degrees.
    rotation_axis : {'x', 'y', 'z'}
        The axis of rotation.
    clim : list of float
        The color range limits.
    ambient : float
        The ambient lighting amount
    save_path : str
        The path to save the images.

    Returns
    -------
    None

    """

    pl = pv.Plotter(off_screen=True)
    pl.enable_anti_aliasing()
    pl.set_background("white")

    pl.add_mesh(
        mesh=geometry,
        cmap=cm.jet,
        show_scalar_bar=False,
        clim=clim,
        ambient=ambient,
        smooth_shading=True,
        lighting=True
    )

    for i in range(360 // rotation_step):
        if rotation_axis == "x":
            geometry.rotate_x(rotation_step, inplace=True)
        elif rotation_axis == "y":
            geometry.rotate_y(rotation_step, inplace=True)
        elif rotation_axis == "z":
            geometry.rotate_z(rotation_step, inplace=True)
        else:
            raise ValueError("Roatation axis is not correct")

        pl.show(auto_close=False)
        image = Image.fromarray(pl.image[:, 128:-128, :])
        # ... Grayscale Convertion
        # image = image.convert("L")
        image.save(save_path + "_{:s}_{:03d}.png".format(rotation_axis, i))

    pl.close()
    pl.deep_clean()