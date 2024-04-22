import numpy as np
import taichi as ti
from PLY_data import PLY_data

from ply_util import write_ply

def random_rotation_matrix(strength=None, dtype=None):#know
    """Generates a random rotation matrix 
    
    strength: scalar in [0,1]. 1 generates fully random rotations. 0 generates the identity. Default is 1.
    dtype: output dtype. Default is np.float32
    """
    if strength is None:
        strength = 1.0

    if dtype is None:
        dtype = np.float32

    x = np.random.rand(3)
    theta = x[0] * 2 * np.pi * strength
    phi = x[1] * 2 * np.pi
    z = x[2] * strength

    r = np.sqrt(z)
    V = np.array([np.sin(phi) * r, np.cos(phi) * r, np.sqrt(2.0 - z)])

    st = np.sin(theta)
    ct = np.cos(theta)

    Rz = np.array([[ct, st, 0], [-st, ct, 0], [0, 0, 1]])

    rand_R = (np.outer(V, V) - np.eye(3)).dot(Rz)
    return rand_R.astype(dtype)

def Rotate1frame(idx):

    basedir=r"D:\CODE\dataProcessing\\"
    filename=r"rigidx.ply"

    basedir=r"d:\CODE\Tichi_SPH\ply_models\\"
    filename=r"bunny_0.05.ply"

    plydata=PLY_data(
        ply_filename=basedir+filename,
        offset=ti.Vector([0,0,0]))
    fluid=plydata.pos
    print(fluid.shape)

    #ply 2 np 2 ply
    R = random_rotation_matrix(1.0)
    print('[rotate shape]')
    print(fluid.shape)
    fluid=fluid @ R

    write_ply(
            path=basedir,
            frame_num=idx,
            type="solid",
            dim=3,
            num=fluid.shape[0],
            pos=fluid,
            phase_num=1,
            solid_beta=np.zeros_like(fluid)
            )

    
    return

for i in range(0,10):
    Rotate1frame(i)
print('[done]')