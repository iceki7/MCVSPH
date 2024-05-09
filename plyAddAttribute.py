import numpy as np

from plyfile import PlyData, PlyElement


#from    https://github.com/dranjan/python-plyfile/issues/26
prm_overwrite=1


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

def plyaddRGB(filename, attrarray):
    p = PlyData.read(filename)
    v = p.elements[0]   #v[0] has key x,y,z
    # print('vx shape')
    # print(v['x'].shape)
    # f = p.elements[1] # err
    flag_hasrgb=0
    # print(v.data.dtype.descr)

    for oriattr in v.data.dtype.descr:
        if ('red' in oriattr):
            flag_hasrgb=1
            # print('rgb exsited in '+str(filename))
            if(prm_overwrite==0):
                return
            
            break

    # Create the new vertex data with appropriate dtype
    if(flag_hasrgb==0):
        a = np.empty(len(v.data), v.data.dtype.descr + [('red', 'f4')]+ [('green', 'f4')]+ [('blue', 'f4')])
    else:
        a = np.empty(len(v.data), v.data.dtype.descr)

    for name in v.data.dtype.fields:
        a[name] = v[name]
    a['red'] =   attrarray[:,0]
    a['green'] = attrarray[:,1]
    a['blue'] =  attrarray[:,2]
    # Recreate the PlyElement instance
    v = PlyElement.describe(a, 'vertex')

    # Recreate the PlyData instance
    p = PlyData([v], text=True)

    p.write(filename)