import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))

import os
import random
import numpy as np
import pandas as pd
import pyvista as pv
from PIL import Image
from matplotlib.pyplot import cm

random.seed(7)

ROOT_PARENT = r"E:\cfdcfd_DL"
FOLDER_RANGE = range(2, 25)
IMAGES_DIR = "Images"
TRAIN_DIR = "Train_2"
TEST_DIR = "Test_2"
GEOMETRY_TRANSFORMATIONS = ["Raw", "Curvature", "TAWSS", "ECAP", "OSI", "RRT"]
TRAIN_PERCENTAGE = 0.8
ROTATION_STEP = 30

def get_clim(transformation):
    if transformation == "Curvature":
        return [-30, 100.0]
    elif transformation == "TAWSS":
        return [-3.0, 3.0]
    elif transformation == "ECAP":
        return [0.0, 2.0]
    elif transformation == "OSI":
        return [0.0, 0.5]
    elif transformation == "RRT":
        return [0.0, 10]
    else:
        return [0.0, 0.0]

def get_ambient(transformation):
    return 0.1 if transformation == "Raw" else 0.3

def generate_rotating_snapshots(
    geometry, rotation_step, rotation_axis, clim, ambient, save_path, case_name
):
    pl = pv.Plotter(off_screen=True, window_size=[768, 768])
    pl.enable_anti_aliasing()
    pl.set_background("white")
    for i in range(360 // rotation_step):
        geom_copy = geometry.copy(deep=True)
        angle = rotation_step * i
        if rotation_axis == "x":
            geom_copy.rotate_x(angle, inplace=True)
            view_vector = (1, 0, 0)
        elif rotation_axis == "y":
            geom_copy.rotate_y(angle, inplace=True)
            view_vector = (0, 1, 0)
        elif rotation_axis == "z":
            geom_copy.rotate_z(angle, inplace=True)
            view_vector = (0, 0, 1)
        pl.clear()
        pl.add_mesh(
            mesh=geom_copy,
            cmap=cm.jet,
            show_scalar_bar=False,
            clim=clim,
            ambient=ambient,
            smooth_shading=True,
            lighting=True
        )
        # 카메라 시점을 축별로 맞춤
        pl.camera_position = 'xy'  # 기본값, 아래에서 바꿈
        if rotation_axis == "x":
            pl.view_vector(view_vector, viewup=(0, 0, 1))
        elif rotation_axis == "y":
            pl.view_vector(view_vector, viewup=(0, 0, 1))
        elif rotation_axis == "z":
            pl.view_vector(view_vector, viewup=(0, 1, 0))
        pl.enable_eye_dome_lighting()
        pl.view_isometric()
        pl.reset_camera()
        image = Image.fromarray(pl.screenshot(return_img=True)[..., :3])
        image.save(f"{save_path}_{rotation_axis}_{i:03d}.png")
    pl.close()

def generate_images_from_geometry(
    geometry_path, cfd_path, transformation, save_dir, case_name
):
    geometry = pv.read(geometry_path)
    cfd_results = pd.read_csv(cfd_path)
    if transformation == "Raw":
        pass
    elif transformation == "Curvature":
        curvature = geometry.curvature(curv_type="gaussian")
        curvature[curvature < 0.001] = 0.001
        geometry.point_data[transformation] = np.log2(curvature)
    elif transformation == "TAWSS":
        geometry.point_data[transformation] = np.log2(
            cfd_results.filter(regex=f".*{transformation}.*"))
    else:
        geometry.point_data[transformation] = cfd_results.filter(
            regex=f".*{transformation}.*")
    # z축(원본 방향) 360도 회전 12장
    generate_rotating_snapshots(
        geometry=geometry,
        rotation_step=ROTATION_STEP,
        rotation_axis="z",
        clim=get_clim(transformation),
        ambient=get_ambient(transformation),
        save_path=os.path.join(save_dir, case_name),
        case_name=case_name
    )
    # x축 기준 360도 회전 12장
    generate_rotating_snapshots(
        geometry=geometry,
        rotation_step=ROTATION_STEP,
        rotation_axis="x",
        clim=get_clim(transformation),
        ambient=get_ambient(transformation),
        save_path=os.path.join(save_dir, case_name),
        case_name=case_name
    )
    # y축 기준 360도 회전 12장
    generate_rotating_snapshots(
        geometry=geometry,
        rotation_step=ROTATION_STEP,
        rotation_axis="y",
        clim=get_clim(transformation),
        ambient=get_ambient(transformation),
        save_path=os.path.join(save_dir, case_name),
        case_name=case_name
    )

if __name__ == "__main__":
    all_cases = []
    for folder_num in FOLDER_RANGE:
        folder_path = os.path.join(ROOT_PARENT, str(folder_num))
        if not os.path.isdir(folder_path):
            continue
        for date_folder in os.listdir(folder_path):
            date_path = os.path.join(folder_path, date_folder)
            wss_path = os.path.join(date_path, "wss")
            stl_file = os.path.join(wss_path, f"{date_folder}.stl")
            cfd_file = os.path.join(wss_path, "hemodynamics_tawss_osi_rrt_ecap.csv")
            if os.path.exists(stl_file) and os.path.exists(cfd_file):
                case_name = f"{folder_num}_{date_folder}"
                all_cases.append((case_name, wss_path, stl_file, cfd_file))

    random.shuffle(all_cases)
    split_idx = int(len(all_cases) * TRAIN_PERCENTAGE)
    train_cases = all_cases[:split_idx]
    test_cases = all_cases[split_idx:]

    for transformation in GEOMETRY_TRANSFORMATIONS:
        for case_name, wss_path, stl_path, cfd_path in train_cases:
            train_output_dir = os.path.join(ROOT_PARENT, IMAGES_DIR, TRAIN_DIR, transformation)
            os.makedirs(train_output_dir, exist_ok=True)
            generate_images_from_geometry(
                geometry_path=stl_path,
                cfd_path=cfd_path,
                transformation=transformation,
                save_dir=train_output_dir,
                case_name=case_name
            )
        for case_name, wss_path, stl_path, cfd_path in test_cases:
            test_output_dir = os.path.join(ROOT_PARENT, IMAGES_DIR, TEST_DIR, transformation)
            os.makedirs(test_output_dir, exist_ok=True)
            generate_images_from_geometry(
                geometry_path=stl_path,
                cfd_path=cfd_path,
                transformation=transformation,
                save_dir=test_output_dir,
                case_name=case_name
            )
