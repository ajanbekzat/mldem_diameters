"""Helper functions for reading and writing simulation data"""
import os
import re
from glob import glob

import numpy as np
import vtk


def mfixomg_numpy_from_bgeo(path):
    # create reader
    points = vtk.vtkXMLPolyDataReader()
    points.SetFileName(path)
    points.Update()
    # print the arrays
    data = points.GetOutput()
    point_data = data.GetPointData()
    n = data.GetNumberOfPoints()
    pos_arr = np.empty((n, 3))
    vel_arr = np.empty((n, 3))
    omg_arr = np.empty((n, 3))
    ori_arr = np.empty((n, 3))
    rho_arr = np.empty((n))
    # loop over all data arrays
    for i in range(point_data.GetNumberOfArrays()):
        #print(i,point_data.GetArrayName(i))
        if(point_data.GetArrayName(i)=='Density'):
            rou=point_data.GetArray(i)        
        if(point_data.GetArrayName(i)=='Velocity'):
            vel=point_data.GetArray(i)
        if(point_data.GetArrayName(i)=='Angular_velocity'):
            omg=point_data.GetArray(i)
        if(point_data.GetArrayName(i)=='Orientation'):
            ori=point_data.GetArray(i)
            
    #get xyz
    for i in range(n):
        p = data.GetPoints().GetPoint(i) #p is a tuple with the x,y & z coordinates.
        v = [vel.GetValue(3*i),vel.GetValue(3*i+1),vel.GetValue(3*i+2)]
        w = [omg.GetValue(3*i),omg.GetValue(3*i+1),omg.GetValue(3*i+2)]
        o = [ori.GetValue(3*i),ori.GetValue(3*i+1),ori.GetValue(3*i+2)]
        rho = rou.GetValue(i)
        pos_arr[i] = p
        vel_arr[i] = v
        omg_arr[i] = w
        ori_arr[i] = o
        rho_arr[i] = rho
        #if i==0:
        #    print(p,v,rho)

    result = [pos_arr, vel_arr, omg_arr,ori,rho_arr]
    return tuple(result)

def mfix_numpy_from_bgeo(path):
    # create reader
    points = vtk.vtkXMLPolyDataReader()
    points.SetFileName(path)
    points.Update()
    # print the arrays
    data = points.GetOutput()
    point_data = data.GetPointData()
    n = data.GetNumberOfPoints()
    diam_arr = np.empty((n))
    pos_arr = np.empty((n, 3))
    vel_arr = np.empty((n, 3))
    rho_arr = np.empty((n))
    # loop over all data arrays
    for i in range(point_data.GetNumberOfArrays()):
        feat_name = point_data.GetArrayName(i)
        # print(feat_name)
        if(feat_name=='Diameter' or feat_name=='radius'):
            diam=point_data.GetArray(i)
        if(feat_name == 'Density' or feat_name=='density'):
            rou=point_data.GetArray(i)
        if(feat_name=='Velocity' or feat_name=='v'):
            vel=point_data.GetArray(i)
  
    #get xyz
    for i in range(n):
        diam_1 = diam.GetValue(i)
        p = data.GetPoints().GetPoint(i) #p is a tuple with the x,y & z coordinates.
        v = [vel.GetValue(3*i),vel.GetValue(3*i+1),vel.GetValue(3*i+2)]
        rho = rou.GetValue(i)
        diam_arr[i] = diam_1
        pos_arr[i] = p
        vel_arr[i] = v
        rho_arr[i] = rho
        #if i==0:
        #    print(p,v,rho)

    result = [diam_arr, pos_arr, vel_arr, rho_arr]
    return tuple(result)

def mfix_get_fluid_frame_id_from_bgeo_path(x):
    return int(re.match('.*PARTICLEDATA_.+_(\d+)\.vtp', x).group(1))


def mfix_get_fluid_ids_from_partio_dir(partio_dir):
    bgeo_files = glob(os.path.join(partio_dir, 'PARTICLEDATA*.vtp'))
    fluid_ids = set()
    for x in bgeo_files:
        fluid_ids.add(re.match('.*PARTICLEDATA_(.+)_\d+\.vtp', x).group(1))

    return list(sorted(fluid_ids))


def mfix_get_fluid_bgeo_files(partio_dir, fluid_id):
    bgeo_files = glob(
        os.path.join(partio_dir, 'PARTICLEDATA_{0}_*.vtp'.format(fluid_id)))
    bgeo_files.sort(key=mfix_get_fluid_frame_id_from_bgeo_path)
    return bgeo_files

def get_fluid_frame_id_from_bgeo_path(x):
    return int(re.match('.*ParticleData_.+_(\d+)\.bgeo', x).group(1))


def get_fluid_ids_from_partio_dir(partio_dir):
    bgeo_files = glob(os.path.join(partio_dir, 'ParticleData*.bgeo'))
    fluid_ids = set()
    for x in bgeo_files:
        fluid_ids.add(re.match('.*ParticleData_(.+)_\d+\.bgeo', x).group(1))

    return list(sorted(fluid_ids))


def get_fluid_bgeo_files(partio_dir, fluid_id):
    bgeo_files = glob(
        os.path.join(partio_dir, 'ParticleData_{0}_*.bgeo'.format(fluid_id)))
    bgeo_files.sort(key=get_fluid_frame_id_from_bgeo_path)
    return bgeo_files


def numpy_from_bgeo(path):
    import partio
    p = partio.read(path)
    pos = p.attributeInfo('position')
    vel = p.attributeInfo('velocity')
    ida = p.attributeInfo('trackid')  # old format
    if ida is None:
        ida = p.attributeInfo('id')  # new format after splishsplash update
    n = p.numParticles()
    pos_arr = np.empty((n, pos.count))
    for i in range(n):
        pos_arr[i] = p.get(pos, i)

    vel_arr = None
    if not vel is None:
        vel_arr = np.empty((n, vel.count))
        for i in range(n):
            vel_arr[i] = p.get(vel, i)

    if not ida is None:
        id_arr = np.empty((n,), dtype=np.int64)
        for i in range(n):
            id_arr[i] = p.get(ida, i)[0]

        s = np.argsort(id_arr)
        result = [pos_arr[s]]
        if not vel is None:
            result.append(vel_arr[s])
    else:
        result = [pos_arr, vel_arr]

    return tuple(result)


def write_bgeo_from_numpy(outpath, pos_arr, vel_arr):
    import partio

    n = pos_arr.shape[0]
    if not (vel_arr.shape[0] == n and pos_arr.shape[1] == 3 and
            vel_arr.shape[1] == 3):
        raise ValueError(
            "invalid shapes for pos_arr {} and/or vel_arr {}".format(
                pos_arr.shape, vel_arr.shape))

    p = partio.create()
    position_attr = p.addAttribute("position", partio.VECTOR, 3)
    velocity_attr = p.addAttribute("velocity", partio.VECTOR, 3)

    for i in range(n):
        idx = p.addParticle()
        p.set(position_attr, idx, pos_arr[i].astype(float))
        p.set(velocity_attr, idx, vel_arr[i].astype(float))

    partio.write(outpath, p)
