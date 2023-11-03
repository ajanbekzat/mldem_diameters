# 1st script
#!/usr/bin/env python3
"""This script creates compressed records for training the network"""
import argparse
import json
import os
import sys
from glob import glob

import numpy as np
import open3d as o3d
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

    box, box_normals = stl_to_particles(os.path.join(scene_dir, "geometry.stl"))

    partio_dir = os.path.join(scene_dir, "PARTIO")
    fluid_ids = ["FLUID0"]  # mfix_get_fluid_ids_from_partio_dir(partio_dir)
    # print(fluid_ids)
    num_fluids = len(fluid_ids)
    fluid_id_bgeo_map = {k: mfix_get_fluid_bgeo_files(partio_dir, k) for k in fluid_ids}

    frames = None

    for k, v in fluid_id_bgeo_map.items():
        if frames is None:
            frames = list(range(len(v)))
        if len(v) != len(frames):
            raise Exception(
                "number of frames for fluid {} ({}) is different from {}".format(
                    k, len(v), len(frames)
                )
            )
    if frames is None:
        return

    print("--------------Processing------{:} frames={:}".format(scene_dir, len(frames)))
    #    print("-----------------box={:}".format(box.shape[:]))
    sublists = np.array_split(frames, splits)

    boring = False  # no fluid and rigid bodies dont move
    last_max_velocities = [1] * 20

    for sublist_i, sublist in enumerate(sublists):
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

                pos = []
                vel = []
                mass = []
                viscosity = []

                sizes = np.array([0, 0, 0, 0], dtype=np.int32)

                for flid in fluid_ids:
                    bgeo_path = fluid_id_bgeo_map[flid][frame_i]
                    pos_, vel_, density_ = mfix_numpy_from_bgeo(bgeo_path)
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

                pos = np.concatenate(pos, axis=0)
                vel = np.concatenate(vel, axis=0)
                mass = np.concatenate(mass, axis=0)
                mass *= (2 * PARTICLE_RADIUS) ** 3
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

                data.append(feat_dict)

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
    parser = argparse.ArgumentParser(
        description="Creates compressed msgpacks for directories with SplishSplash scenes"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="The path to the output directory"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="The path to the input directory with the simulation data",
    )
    parser.add_argument(
        "--splits",
        type=int,
        default=1,
        help="The number of files to generate per scene (default=16)",
    )

    args = parser.parse_args()
    os.makedirs(args.output)

    outdir = args.output
    print(outdir)

    scene_dirs = sorted(glob(os.path.join(args.input, "*")))
    print(scene_dirs)

    for scene_dir in scene_dirs:
        print(scene_dir)
        scene_name = os.path.basename(scene_dir)
        print(scene_name)
        outfileprefix = os.path.join(outdir, scene_name)
        create_scene_files(scene_dir, scene_name, outfileprefix, args.splits)

    print("end")


if __name__ == "__main__":
    main()
