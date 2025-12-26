# prm_
import numpy as np
import tqdm

from get1ply import get1ply

prm_average_vel=1

##用平均速度计算能量

lv=0
rv=999
solvername="df"
prefix=r"D:\CODE\MCVSPH-FORK\specific_"+solvername+"_1_output\\"


a_avvel=[]
a_avvor=[]
avel=[]
dt_frame=0.016
for i in tqdm.tqdm(range(lv,rv+1)):



    if(i%20==0):
        print(i)

    if(prm_average_vel):
        pos0=get1ply(prefix+r"particle_object_0_"+str(i)+".ply")
        pos1=get1ply(prefix+r"particle_object_0_"+str(i+1)+".ply")
        vel=(pos1-pos0)/dt_frame
        
        avel.append(np.mean(vel**2))


np.save(prefix+"av_ene_"+solvername+"_cp",avel)