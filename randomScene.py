

import numpy as np
from boxNorm import zxcboxandnorm
from config_builder import SimConfig
import json
import os

np.random.seed(1234)






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



        de[dim]=np.random.uniform(de[dim]*0.3,de[dim]*1.3)

        dsize=de[dim]-ds[dim]
        dcenter=(de[dim]+ds[dim])/2

        fsize=np.random.uniform(dsize*0.2,dsize*0.6)

        if(dim==1):#y
            fsize=np.random.uniform(dsize*0.3,dsize*0.5)


        data["FluidBlocks"][0]["start"][dim]=dcenter-fsize/2
        data["FluidBlocks"][0]["end"]  [dim]=dcenter+fsize/2

        data["FluidBlocks"][0]["translation"][dim]=np.random.uniform(-0.3,0.3)


        fe=data["FluidBlocks"][0]["end"]  [dim]+data["FluidBlocks"][0]["translation"][dim]
        fs=data["FluidBlocks"][0]["start"][dim]+data["FluidBlocks"][0]["translation"][dim]
        

        if(de[dim]-fe<0.03):
            data["FluidBlocks"][0]["end"]  [dim]=de[dim]-0.1-data["FluidBlocks"][0]["translation"][dim]

        if(fs-ds[dim]<0.03):
            data["FluidBlocks"][0]["start"][dim]=ds[dim]+0.1-data["FluidBlocks"][0]["translation"][dim]


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


    
    
    boxp,boxn=zxcboxandnorm(
        lb=np.array(ds),
        rt=np.array(de),
        rad=data["Configuration"]["particleRadius"])
    np.save("randomScene/bp-"+baseScene.split(".")[0]+"-r"+str(idx),boxp)
    np.save("randomScene/bn-"+baseScene.split(".")[0]+"-r"+str(idx),boxn)


num=100
baseScene="lowfluid.json"#prm

baseScene="lowfluid-S.json"#prm

#zxc 生成随即场景json

for i in range(num):
    get1(baseScene,i)

print('[av part num]')
print(avpartnum/num)