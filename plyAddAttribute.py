import numpy as np

from plyfile import PlyData, PlyElement


#from    https://github.com/dranjan/python-plyfile/issues/26

def plyaddattr(filename, attrarray,attrname):
    p = PlyData.read(filename)
    v = p.elements[0]   #v[0] has key x,y,z
    # print('vx shape')
    # print(v['x'].shape)
    # f = p.elements[1] # err

    # print(v.data.dtype.descr)
    for oriattr in v.data.dtype.descr:
        if (attrname in oriattr):
            print('existed in '+str(filename))
            return

    # Create the new vertex data with appropriate dtype
    a = np.empty(len(v.data), v.data.dtype.descr + [(attrname, 'f4')])
    for name in v.data.dtype.fields:
        a[name] = v[name]
    a[attrname] = attrarray

    # Recreate the PlyElement instance
    v = PlyElement.describe(a, 'vertex')

    # Recreate the PlyData instance
    p = PlyData([v], text=True)

    p.write(filename)