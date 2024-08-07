
#速度场P2G

import taichi as ti
from plyfile import *
from tqdm import tqdm
from get1ply import get1ply

first=1


def write_ply(path, frame_num,dim, num, pos,usevel=0,vel=0,replacevel=0):
    if dim == 3:
        list_pos = []
        for i in range(num):
            pos_tmp = [pos[i, 0], pos[i, 1], pos[i, 2]]
            list_pos.append(tuple(pos_tmp))
    elif dim == 2:
        list_pos = [(pos[i, 0], pos[i, 1], 0) for i in range(num)]
    else:
        print('write_ply(): dim exceeds default values')
        return
    data_type = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
 
    np_pos = np.array(list_pos, dtype=data_type)
    el_pos = PlyElement.describe(np_pos, 'vertex')
    PlyData([el_pos]).write(str(path) +'{0:04d}.ply'.format(frame_num))



# https://www.sidefx.com/docs/houdini//nodes/sop/volumerasterizeparticles.html
# node

#zxc
# partvel 2 gridvel
# 1.遍历网格
# 2.对每个网格，遍历周围的粒子，
# 3.速度加权求和 vi=W vp

ti.init(arch=ti.gpu,
         device_memory_GB=1, 
         debug=False,
         random_seed=int(1234),kernel_profiler=False)





# gridpos


import numpy as np  
import matplotlib.pyplot as plt  

#prm_------------------------------------




prm_loadnpz=0
prm_self=0#分解自身
prm_high=1#高频
prm_acc=0 #累计矫正

ratio=0.43

gridsize=0.8
# gridsize=0.375

prm_readEnhanceVel=0
prm_onlyp2g=1
gridsize=1;equalaxis=1
# gridsize=0.4;  equalaxis=0
# gridsize=0.1;   equalaxis=1
gridsize=0.05;   equalaxis=1


dt_frame=0.016

particlemaxnum=50000
dim=3






prefix=r"D:\\CODE\\CCONV RES\\csm_mp300_50kexample_long2z+2\\"
domain_start=np.array([-1,  0,      -4.7])
domain_end=  np.array([1,   2.6,    4])
slicey=3
prm_lv=1
prm_rv=1000





prefix=r"D:/CODE/CCONV RES/mix/y0_csm_mp300emit_large/y0_csm_mp300emit_large/"
domain_start=np.array([-14.4, -4.2,      -2.9,])
domain_end=  np.array([8.4,   -3.,    2.9,])
slicey=3
prm_lv=566
prm_rv=566
formatnum=1
filepre=r"fluid_"



# prefix=r"D:/CODE/CCONV RES/mix/_csm300_1111_50kmc_ball_2velx_0602/_csm300_1111_50kmc_ball_2velx_0602/"
# prefix=r"D:/CODE/CCONV RES/mix/_csm_df300_1111_50kmc_ball_2velx_0602/_csm_df300_1111_50kmc_ball_2velx_0602/"
# prefix=r'D:/CODE/CCONV RES/mix/emin_b_mc_ball_2velx_0602/emin_b_mc_ball_2velx_0602/'
# # prefix=r'D:/CODE/MCVSPH-FORK/specific_mt_1_output/'
# # prefix=r'D:/CODE/MCVSPH-FORK/specific_df_1_output/'
# # prefix=r'D:/CODE/MCVSPH-FORK/specific_mp_1_output/'
# domain_start=np.array([-17, -4.2,      -0.8,])
# domain_end=  np.array([8.5,   -2.3,    0.8,])
# slicey=3
# prm_lv=1
# prm_rv=1000
# formatnum=1
# # filepre=r"particle_object_0_"
# filepre=r"fluid_"


#--------------------------



domain_size=domain_end-domain_start

gridnumx=int(domain_size[0]/gridsize)+1
gridnumy=int(domain_size[1]/gridsize)+1

gridnum=gridnumx*gridnumy
if(dim==3):
    gridnumz=int(domain_size[2]/gridsize)+1
    gridnum*=gridnumz

#邻域半径
support_radius=gridsize

particlenum_leminar=0
particlenum0=0

X="none"
Y="none"
Z="none"
def initGridCoord():
    global X,Y,Z
    x = np.linspace(domain_start[0], domain_end[0], gridnumx)  # 创建一个包含domain_end0个元素的等差数列，从domain_start到domain_end  
    y = np.linspace(domain_start[1], domain_end[1], gridnumy)  # 同上  
    if(dim==3):
        z = np.linspace(domain_start[2], domain_end[2], gridnumz)  # 同上  
    # 使用meshgrid生成坐标矩阵  
    if(dim==3):
        X, Y ,Z = np.meshgrid(x,y,z)  
        ZF=Z.flatten()
    else:
        X, Y  = np.meshgrid(x,y)  
    XF=X.flatten()
    YF=Y.flatten()

    # for i in range(0,X.shape[0]):
    #     print([int(X[i]),int(Y[i]),int(Z[i])])

    if(dim==3):    
        coord=np.array([XF,YF,ZF]).T
    else:
        coord=np.array([XF,YF]).T

    print('[grid num]'+str(gridnum))
    return coord

gridneighbornum=  ti.field(dtype=int, shape=gridnum)
partneighbornum=  ti.field(dtype=float, shape=particlemaxnum)

gridpos=  ti.Vector.field(dim, dtype=float, shape=gridnum)
gridvel=  ti.Vector.field(dim, dtype=float, shape=gridnum)
gridvelnorm=  ti.field(dtype=float, shape=gridnum)



partpos0=  ti.Vector.field(dim, dtype=float, shape=particlemaxnum)
partvel0=  ti.Vector.field(dim, dtype=float, shape=particlemaxnum)#用于增强
partvel_tq=ti.Vector.field(dim, dtype=float, shape=particlemaxnum)#提取速度

partpos_leminar=  ti.Vector.field(dim, dtype=float, shape=particlemaxnum)
partvel_leminar=  ti.Vector.field(dim, dtype=float, shape=particlemaxnum)

partvel_export =  ti.Vector.field(dim, dtype=float, shape=particlemaxnum)
partpos_accumulate=  ti.Vector.field(dim, dtype=float, shape=particlemaxnum)



gridpos.from_numpy(initGridCoord())

@ti.data_oriented
class velInterpolate:

    #遍历邻域中的粒子
    @ti.func
    def for_all_Gsneighbors(self, g_i, task: ti.template(), ret: ti.template()):

        for p_j in range(0,particlenum0):
            if (gridpos[g_i] - partpos0[p_j]).norm() < support_radius:
                # print('[pair]')
                # print(g_i)
                # print(p_j)
                ti.atomic_add(gridneighbornum[g_i], 1)
                task(g_i, p_j, ret)
    @ti.func
    def for_all_Psneighbors(self, p_i, task: ti.template(), ret: ti.template()):

        for g_j in range(0,gridnum):
            if (partpos_leminar[p_i] - gridpos[g_j]).norm() < support_radius:
                ti.atomic_add(partneighbornum[p_i], 1)
                task(p_i, g_j, ret)   

    @ti.func
    def cubic_kernel(self, r_norm):
        res = ti.cast(0.0, ti.f32)
        h = support_radius
        # value of cubic spline smoothing kernel

        k = 8 / np.pi
        k /= h ** dim
        q = r_norm / h
        if q <= 1.0:
            if q <= 0.5:
                q2 = q * q
                q3 = q2 * q
                res = k * (6.0 * q3 - 6.0 * q2 + 1)
            else:
                res = k * 2 * ti.pow(1 - q, 3.0)
        return res


    @ti.func
    def velP2G_task(self, g_i, p_j,
                                            curl_v: ti.template()):
        # gridvel[g_i]+=partvel0[p_j]
        gridvel[g_i]+=partvel0[p_j]*self.cubic_kernel((gridpos[g_i] - partpos0[p_j]).norm())

            # (gridpos[g_i] - partpos[p_j]).norm()
            # self.cubic_kernel_derivative(gridpos[g_i] - partpos[p_j])

    @ti.func
    def velG2P_task(self, p_i, g_j,
                                            curl_v: ti.template()):
        # partvel_tq[p_i]+=gridvel[g_j]
        partvel_tq[p_i]+=gridvel[g_j]*self.cubic_kernel(( partpos_leminar[p_i]-gridpos[g_j]).norm() )



   
    @ti.kernel
    def velP2G(self):
        for g_i in ti.grouped(gridpos):
            v_curl = ti.Vector([0.0, 0.0, 0.0])
            gridvel[g_i]=ti.Vector([0.0, 0.0, 0.0])
            gridneighbornum[g_i]=0

            self.for_all_Gsneighbors(
                g_i, self.velP2G_task, v_curl)
            # print('[gridpartnum]')
            # print(gridneighbornum[g_i])
            gridvel[g_i]/=gridneighbornum[g_i]
            # a=ti.Vector([3,4,12])
            # print('norm')
            # print(a.norm())
            
            gridvelnorm[g_i]=gridvel[g_i].norm()
            # print(gridvel[g_i])

    @ti.kernel
    def velG2P(self):
        v_curl = ti.Vector([0.0, 0.0, 0.0])
        for p_i in ti.grouped(partpos_leminar):

            partvel_tq[p_i]=ti.Vector([0.0, 0.0, 0.0])
            partneighbornum[p_i]=0

            self.for_all_Psneighbors(
                p_i, self.velG2P_task, v_curl)
            # print('[partpartnum]')
            # print(partneighbornum[p_i])
            partvel_tq[p_i]/=partneighbornum[p_i]
            if(prm_high):
                partvel_tq[p_i]=partvel0[p_i]-partvel_tq[p_i]
            # partvel[p_i]*=ratio
            # print(gridvel[g_i])
                        

#覆盖partvel,partpos
def loadfile(filename,idx):

    global particlenum_leminar
    global particlenum0
    global partpos0

    if(prm_loadnpz):
        data = np.load(filename)


    if(prm_readEnhanceVel):

        if(prm_loadnpz):
            partpos0.from_numpy(data['pos'])
            partvel0.from_numpy(data['vel'])
            particlenum0=data['pos'].shape[0]
        else:
            if(formatnum):
                pos0=get1ply(prefix+filepre+'{0:04d}.ply'.format(idx))#,extraattr='curlabs')
                pos1=get1ply(prefix+filepre+'{0:04d}.ply'.format(idx+1))#,extraattr='curlabs')

            else:
                pos0=get1ply(prefix+filepre+str(idx)+'.ply')
                pos1=get1ply(prefix+filepre+str(idx+1)+'.ply')

            partpos0.from_numpy(pos0)
            # partpos0.from_numpy(pos0[:,:3])

            partvel0.from_numpy((pos1-pos0)/dt_frame)
            # partvel0.from_numpy(np.array([pos0[:,3]]).T)
       
            particlenum0=pos0.shape[0]

            print('[part vel]')
            print(np.mean((pos1-pos0)/dt_frame))
            print(np.max((pos1-pos0)/dt_frame))
            print(np.min((pos1-pos0)/dt_frame))
            print('[partnum]\t'+str(pos0.shape))



       

    else:
        particlenum_leminar=data['pos'].shape[0]
        partvel_leminar.from_numpy(data['vel'])
        global first
        if(prm_acc):
            if(first):
                first=0
                partpos_accumulate.from_numpy(data['pos'])

        else:
            partpos_accumulate.from_numpy(data['pos'])
            


#prm 


cnt=0






# #将上采样后的细节加入下述模拟中
# prefix2=r"D:\\CODE\\CCONV RES\\pretrained_model_weights_50kexample_long2z+2\\"
# filepre2=r"fluid_"


#
if(prm_self):
    prefix2=prefix
    filepre2=filepre

testname="cubic"
testname="cubicNoacc"

#截取三维网格的一个截面，绘图，
def slice(data,idx,slicey=1,flt=0):
    # print(data.shape)
    data=data[slicey,:,:]
    data[np.isnan(data)]=0
    data=data.T
    plt.imshow(data, cmap='hot', \
                # interpolation='gaussian',\
                vmax=13000,
                vmin=0,
                 )
    plt.colorbar()
    if(flt==0):
        plt.savefig(prefix+'slice-'+str(idx)+'-y'+str(slicey)+'.png',bbox_inches='tight',pad_inches=0.0, dpi=300)
    else:
        plt.savefig(prefix+'slice-'+str(idx)+'-flt-y'+str(slicey)+'.png',bbox_inches='tight',pad_inches=0.0, dpi=300)
    
    plt.close()



    
def quiver(x,y,z,vel,velnorm,idx):
    import matplotlib.pyplot as plt
    import numpy as np
    global slicey
    # print(vel.shape)#grid num 3
    # print(velnorm.shape)#grid num

    velnorm=np.reshape(velnorm,x.shape)

    if(prm_lv==prm_rv):
        for ii in range(0,gridnumy):
            slice(velnorm,idx=idx,slicey=ii)



   




    #REF     https://matplotlib.org/3.4.2/gallery/mplot3d/quiver3d.html
    #  ax = plt.figure().add_subplot(projection='3d')
    # ax.quiver(x, z, y, np.reshape(vel[:,0],x.shape),\
    #                    np.reshape(vel[:,2],x.shape),\
    #                    np.reshape(vel[:,1],x.shape),\
    #                      length=0.1, normalize=True)
    # ax.set_xlim3d(domain_start[0],domain_end[0])
    # ax.set_ylim3d(domain_start[2],domain_end[2])
    # ax.set_zlim3d(domain_start[1],domain_end[1])
    # if(equalaxis):
    #     ax.set_aspect('equal')#know
    # plt.savefig(prefix+'quiver-'+str(idx)+'.png',bbox_inches='tight',pad_inches=0.0, dpi=300)
    # plt.close()

   
    velnorm[np.isnan(velnorm)] = 0 
    velnorm[np.isinf(velnorm)] = 0 

    # print(velnorm)
    print(np.max(velnorm))
    print('[velnorm mean]'+str(np.mean(velnorm)))


    np.save(prefix+"vel-mat-"+str(idx),velnorm)
    # write_ply(prefix,frame_num=idx,dim=3,num=velnorm.shape[0],pos=velnorm)





for i in tqdm(range(prm_lv,prm_rv+1,1)):


    #1、提取速度场
    prm_readEnhanceVel=1
    #change partvel,partpos
    loadfile((prefix+filepre+'{0:04d}'+r".npz").format(i),idx=i)
    # print('l1')
    # print(particlenum)


    obj1=velInterpolate()

    #上采样到网格上 change gridpos
    obj1.velP2G()
    gridvelnp=gridvel.to_numpy()
    gridvelnormnp=gridvelnorm.to_numpy()
    gridneighbornumnp=gridneighbornum.to_numpy()


    np.save(prefix+"pos-arr"+str(i),gridpos.to_numpy())


    
  


    write_ply(path=prefix+"vel3-arr",frame_num=i,dim=3,num=gridvelnp.shape[0],pos=gridvelnp)
    write_ply(path=prefix+"pos-arr",frame_num=i,dim=3,num=gridpos.to_numpy().shape[0],pos=gridpos.to_numpy())


 


    cnt+=1
    if(prm_onlyp2g):
        quiver(X,Y,Z,gridvelnp,velnorm=gridvelnormnp,idx=i)
        continue
    

    prm_readEnhanceVel=0
    loadfile((prefix2+filepre2+'{0:04d}'+r".npz").format(i))
    # print('l2')
    # print(particlenum)
    #转换回粒子 change partvel
    obj1.velG2P()


    partvel_tq_=     partvel_tq.to_numpy()[0:particlenum_leminar,:]
    partvel_leminar_=partvel_leminar.to_numpy()[0:particlenum_leminar,:]
    partpos_accumulate_=partpos_accumulate.to_numpy()[0:particlenum_leminar,:]
    partpos_export_=    partpos_accumulate.to_numpy()[0:particlenum_leminar,:]


    if(prm_self):
        partpos_accumulate_+=\
            dt_frame*partvel_tq_
    else:
        partpos_accumulate_+=\
            dt_frame*(partvel_leminar_+ partvel_tq_*ratio)
        
    if(prm_acc==0):
        partpos_export_+=partvel_tq_*dt_frame
    partpos_accumulate.from_numpy(partpos_accumulate_)
    partvel_export_=partvel_leminar_+partvel_tq_*ratio



    

    # print(partpos_.shape)
    # exit(0)
    np.savez("D:\\CODE\\MCVSPH-FORK\\recoveprm_rvel\\"+testname+"{0:04d}.npz".format(i),
             pos=partpos_export_,
             vel=partvel_export_)
    write_ply(
        path=r"./recoveprm_rvel/"+testname,
        frame_num=i,
        
        dim=3,
        num=particlenum_leminar,
        pos=partpos_export_,
  
        )
    




