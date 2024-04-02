

import numpy as np
from boxNorm import zxcboxandnorm
from config_builder import SimConfig
import json
import os
np.random.seed(1234)



# 如果碰到边界，那就截断



def get1(file,idx):

    with open("./data/scenes/"+baseScene, 'r') as file:
        data = json.load(file)

    ds=data["Configuration"]["domainStart"]
    de=data["Configuration"]["domainEnd"]



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



    partnum=1
    for dim in range(0,3):
        partnum*=(
                data["FluidBlocks"][0]["end"][dim]-\
                data["FluidBlocks"][0]["start"][dim])/  \
                    (2*data["Configuration"]["particleRadius"])+1
    print('[parnum]\t'+str(partnum))



    # 保存修改后的 JSON 文件
    with open("randomScene/"+baseScene.split(".")[0]+"-r"+str(idx)+".json", 'w') as file:
        json.dump(data, file,indent=4)



    
    boxp,boxn=zxcboxandnorm(
        lb=np.array(ds),
        rt=np.array(de),
        rad=data["Configuration"]["particleRadius"])
    np.save("randomScene/bp-"+baseScene.split(".")[0]+"-r"+str(idx),boxp)
    np.save("randomScene/bn-"+baseScene.split(".")[0]+"-r"+str(idx),boxn)


num=10
baseScene="lowfluid.json"


for i in range(num):
    get1(baseScene,i)
