from plyfile import PlyData, PlyElement
import numpy as np


#chatGPT
def convert_ascii_to_binary_ply(input_file, output_file):
    # 读取 ASCII 编码的 PLY 文件
    ply_data = PlyData.read(input_file)
    
    # 创建用于写入的二进制 PLY 文件结构
    vertex_element = ply_data['vertex']
    
    # 提取顶点数据
    # vertices = np.array([list(vertex) for vertex in vertex_element], dtype=vertex_element.dtype)
    
    # 如果存在面数据，提取面数据
    if 'face' in ply_data:
        face_element = ply_data['face']
        faces = np.array([list(face[0]) for face in face_element], dtype=face_element.dtype)
        face_element = PlyElement.describe(faces, 'face')
        ply_data = PlyData([vertex_element, face_element], text=False)
    else:
        ply_data = PlyData([vertex_element], text=False)
    
    # 写入二进制编码的 PLY 文件
    with open(output_file, 'wb') as f:
        ply_data.write(f)

# 示例调用
basedir="D:\CODE\CCONV RES\csm_mp300_50kmc_ball_2velx_0602\csm_mp300_50kmc_ball_2velx_0602\\"
basedir=r"D:\CODE\CCONV RES\csm300_50kmc_ball_2velx_0602\csm300_50kmc_ball_2velx_0602\\"
lv=1
rv=10
for i in range(lv,rv):
    if(i%20==0):
        print('done '+str(i))
    fname=basedir+'fluid_{0:04d}.ply'.format(i)
    convert_ascii_to_binary_ply(fname,fname)
