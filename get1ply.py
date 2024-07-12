#COPY
import numpy as np


def get1ply(filename):

    from plyfile import PlyData
    import os

    #know 没找到文件会报错
    plydata = PlyData.read(filename)

    vertex =  plydata ['vertex']
       
    x = vertex['x']
    y = vertex['y']
    z = vertex['z']

    combined = np.stack((x, y, z), axis=-1)
    return combined