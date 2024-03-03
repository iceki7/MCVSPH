# Iterate over the files
from plyfile import PlyData
import os
import numpy as np
all_data = []
print(os.getcwd())
filepathj='\\_output\\'
filenamej='vorticity_object_0_00000'
for i in range(5):

    file_path = os.getcwd()+f"{filepathj}"+filenamej+f"{i}.ply" #prm framewithVel_ frameVis10_"
    plydata = PlyData.read(file_path)
    vertex = plydata['vertex']

    # Extracting x, y, z, vx, vy, vz
    x = vertex['x']
    y = vertex['y']
    z = vertex['z']


    # Create a 't' attribute starting at 0.1 and incrementing by 0.1 for each file
    t = np.full((len(x),), fill_value=(i+1) * 0.1)

    # Combine x, y, z, vx, vy, vz, t into one array and append to all_data
    combined = np.stack((x, y, z, t), axis=-1)
    all_data.append(combined)
print('<dt loaded>')    
# Concatenate all data along axis 0 to form a (1000*50, 7) array
all_data_array = np.concatenate(all_data, axis=0)
print(all_data_array.shape)