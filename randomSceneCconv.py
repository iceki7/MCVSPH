

import numpy as np
from boxNorm import zxcboxandnorm
from config_builder import SimConfig
import json
import os
np.random.seed(1234)

boxs=[]


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

R = random_rotation_matrix(1.0)
print('-------R-----------')
print(R)
avpartnum=0
def get1(file,idx):
    
    global avpartnum
    with open("./data/scenes/"+baseScene, 'r') as file:
        data = json.load(file)

    ds=data["Configuration"]["domainStart"]
    de=data["Configuration"]["domainEnd"]


    #基于baseScene进行微调。
    #1  随机设置容器大小
    #2  fluid放置在容器中心
    #3  随机移动容器
    #4  如果碰到边界，截断

    #生成法向量
    #估计粒子数量
    


    for dim in range(0,3):


        bcenter=[0,2,0]
        bsize=[2,4,2]
        bs=[-1,0,-1]
        be=[1,5,1]

        fsize=np.random.uniform(bsize[dim]*0.5,bsize[dim]*0.7)

        if(dim==1):#y
            fsize=np.random.uniform(bsize[dim]*0.1,bsize[dim]*0.3)

        if(dim==1):
            data["FluidBlocks"][0]["start"][dim]=bs[dim]+np.random.uniform(0.3,1.2)
            data["FluidBlocks"][0]["end"][dim]=data["FluidBlocks"][0]["start"][dim]+fsize
        else:
            data["FluidBlocks"][0]["start"][dim]=bcenter[dim]-fsize/2
            data["FluidBlocks"][0]["end"]  [dim]=bcenter[dim]+fsize/2

        if(dim!=1):
            data["FluidBlocks"][0]["start"][dim]+=np.random.uniform(-0.3,0.3)
            data["FluidBlocks"][0]["end"][dim]+=np.random.uniform(-0.3,0.3)

        
      

        if(be[dim]-data["FluidBlocks"][0]["end"][dim]<0.03):
            data["FluidBlocks"][0]["end"]  [dim]=be[dim]-0.1

        if(data["FluidBlocks"][0]["start"][dim]-bs[dim]<0.03):
            data["FluidBlocks"][0]["start"][dim]=bs[dim]+0.1

        

        data["Configuration"]["boxid"]=np.random.choice(["001","002","003",
                                        "004","005","006",
                                        "007","008","009",
                                        "010"])
        
        boxs.append(data["Configuration"]["boxid"])

    #估计粒子数量
    partnum=1
    for dim in range(0,3):
        partnum*=(
                data["FluidBlocks"][0]["end"][dim]-\
                data["FluidBlocks"][0]["start"][dim])/  \
                    (2*data["Configuration"]["particleRadius"])+1
    print('[parnum]\t'+str(partnum))
    avpartnum+=partnum



    # 保存修改后的 JSON 文件
    with open("randomScene/"+baseScene.split(".")[0]+"-r"+str(idx)+".json", 'w') as file:
        json.dump(data, file,indent=4)


    
    



num=100

baseScene="cconv-scene.json"#prm


#zxc 生成随即场景json

for i in range(num):
    get1(baseScene,i)

print('[av part num]')
print(avpartnum/num)

# print(boxs)
realbox=[]
for i in range(0,len(boxs)):
    if(i%3==2):
        realbox.append(boxs[i])
    
np.save("sceneidlist",realbox[:100])
xx=np.load("sceneidlist.npy")
print(xx)
print(xx.shape)