# 1st script
#!/usr/bin/env python3
"""This script creates compressed records for training the network"""
import argparse
import json
import os
import shutil
import sys
from glob import glob

import numpy as np
import open3d as o3d
import pandas as pd
from dotenv import load_dotenv
# from create_physics_scenes import PARTICLE_RADIUS
from physics_data_helper import *

PARTICLE_RADIUS = 0.025


def stl_to_particles(objpath, radius=None):
    if radius is None:
        radius = PARTICLE_RADIUS
    obj = o3d.io.read_triangle_mesh(objpath)
    lluMoreParticle = 1
    particle_area = np.pi * radius**2
    # 1.9 to roughly match the number of points of SPlisHSPlasHs surface sampling
    num_points = int(1.9 * lluMoreParticle * obj.get_surface_area() / particle_area)
    pcd = obj.sample_points_poisson_disk(num_points, use_triangle_normal=True)
    points = np.asarray(pcd.points).astype(np.float32)
    normals = -np.asarray(pcd.normals).astype(np.float32)

    outputPLY = objpath + "{:d}.ply".format(num_points)
    o3d.io.write_point_cloud(outputPLY, pcd)
    print("radius={:}, np={:}".format(radius, num_points))
    return points, normals


def create_scene_files(scene_dir, scene_id, outfileprefix, splits=1):
    #    with open(os.path.join(scene_dir, 'scene.json'), 'r') as f:
    #        scene_dict = json.load(f)

    # Replace your boundaries by particles
    box, box_normals = stl_to_particles(os.path.join(scene_dir, "geometry.stl"))

    # This is where you vtp files are located
    partio_dir = os.path.join(scene_dir, "PARTIO")
    fluid_ids = ["FLUID0"]  # mfix_get_fluid_ids_from_partio_dir(partio_dir)

    # Map to the fluid_ids the vtp file names
    fluid_id_bgeo_map = {k: mfix_get_fluid_bgeo_files(partio_dir, k) for k in fluid_ids}

    frames = None

    for k, v in fluid_id_bgeo_map.items():
        if frames is None:
            frames = list(range(len(v)))
        if len(v) != len(frames):
            raise Exception(
                "number of frames for fluid {} ({}) is different from {}".format(k, len(v), len(frames)))
    if frames is None:
        return

    print("--------------Processing------{:} frames={:}".format(scene_dir, len(frames)))
    #    print("-----------------box={:}".format(box.shape[:]))
    sublists = np.array_split(frames, splits)

    boring = False  # no fluid and rigid bodies dont move
    last_max_velocities = [1] * 20

    for sublist_i, sublist in enumerate(sublists):
        print(sublist_i)
        if boring:
            break
        validCase = True
        outfilepath = outfileprefix + "_{0:02d}.msgpack.zst".format(sublist_i)
        if not os.path.isfile(outfilepath):
            data = []
            np0 = 0
            for frame_i in sublist:
                # llu, only use first 200 frames
                if frame_i > 10000:
                    continue
                # print("-----------------frameID={:}".format(frame_i))
                feat_dict = {}
                # only save the box for the first frame of each file to save memory
                if frame_i == sublist[0]:
                    feat_dict["box"] = box.astype(np.float32)
                    feat_dict["box_normals"] = box_normals.astype(np.float32)

                feat_dict["frame_id"] = np.int64(frame_i)
                feat_dict["scene_id"] = scene_id

                diam = []
                pos = []
                vel = []
                mass = []
                viscosity = []

                sizes = np.array([0, 0, 0, 0], dtype=np.int32)

                for flid in fluid_ids:
                    bgeo_path = fluid_id_bgeo_map[flid][frame_i]
                    diam_, pos_, vel_, density_ = mfix_numpy_from_bgeo(bgeo_path)
                    diam.append(diam_)
                    pos.append(pos_)
                    vel.append(vel_)
                    viscosity.append(
                        np.full(
                            pos_.shape[0:1],
                            # scene_dict[flid]['viscosity'],
                            0.1,
                            dtype=np.float32,
                        )
                    )
                    mass.append(density_)
                    sizes[0] += pos_.shape[0]

                diam = np.concatenate(diam, axis=0)
                pos = np.concatenate(pos, axis=0)
                vel = np.concatenate(vel, axis=0)
                mass = np.concatenate(mass, axis=0)
                # mass = (2 * mass * diam) ** 3
                mass = diam
                # mass *= (2 * PARTICLE_RADIUS) ** 3
                viscosity = np.concatenate(viscosity, axis=0)
                if frame_i == 0:
                    np0 = pos_.shape[:][0]
                else:
                    if pos_.shape[:][0] != np0:
                        validCase = False
                        print(
                            "------------ERROR-----frameID={:d}, np={:d}, np0={:d}".format(
                                frame_i, pos_.shape[:][0], np0
                            )
                        )
                        break

                feat_dict["pos"] = pos.astype(np.float32)
                feat_dict["vel"] = vel.astype(np.float32)
                feat_dict["m"] = mass.astype(np.float32)
                feat_dict["viscosity"] = viscosity.astype(np.float32)
                feat_dict["diam"] = diam.astype(np.float32)
                data.append(feat_dict)

            pos_x, pos_y, pos_z = zip(*feat_dict["pos"])
            vel_x, vel_y, vel_z = zip(*feat_dict["vel"])
            df = pd.DataFrame({
                'PosX': pos_x,
                'PosY': pos_y,
                'PosZ': pos_z,
                'VelX': vel_x,
                'VelY': vel_y,
                'VelZ': vel_z,
                'Diameter': feat_dict["diam"],
                'Mass': feat_dict["m"]
            })
            df.to_csv("blabla.csv")

            if validCase:
                create_compressed_msgpack(data, outfilepath)


def create_compressed_msgpack(data, outfilepath):
    import msgpack
    import msgpack_numpy
    import zstandard as zstd

    msgpack_numpy.patch()

    compressor = zstd.ZstdCompressor(level=22)
    with open(outfilepath, "wb") as f:
        print("writing", outfilepath)
        f.write(compressor.compress(msgpack.packb(data, use_bin_type=True)))


def main():
    # Define the paths directly in the script
    load_dotenv(dotenv_path="../.env")
    output_scenes_dir = os.getenv("OUTPUT_SCENES_DIR")
    output_data_dir = f"{output_scenes_dir}_data"
    splits = 1
    test_scene_names_list = ["S09", "S10"]

    if os.path.exists(output_data_dir):
        shutil.rmtree(output_data_dir)

    os.makedirs(output_data_dir)
    train_model_data = f"{output_data_dir}/train"
    valid_model_data = f"{output_data_dir}/valid"
    os.makedirs(train_model_data)
    os.makedirs(valid_model_data)

    scene_dirs = sorted(glob(os.path.join(output_scenes_dir, "*")))
    print(scene_dirs)

    for scenei, scene_dir in enumerate(scene_dirs):
        scene_name = os.path.basename(scene_dir)
        print(f"Scenei: {scene_name}")
        if scene_name not in test_scene_names_list:
            output_data_trainval = train_model_data
        else:
            output_data_trainval = valid_model_data
        outfileprefix = os.path.join(output_data_trainval, scene_name)
        create_scene_files(scene_dir, scene_name, outfileprefix, splits)

    print("end")


if __name__ == "__main__":
    main()
