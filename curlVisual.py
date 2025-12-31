import argparse
import numpy as np
import taichi as ti
from tqdm import tqdm

from get1ply import get1ply
from plyAddAttribute import plyaddRGB, plyaddattr

#prm_
prefix=r"D:\CODE\CCONV RES\csm_mp200_example_long2z+2\\"
prefix=r"D:\CODE\CCONV RES\csm_mp100_50kexample_long2z+2\\"
prefix=r"D:\CODE\CCONV RES\csm100_50kexample_long2z+2\\"
prefix=r"D:\CODE\CCONV RES\csm_mp200_50kexample_bunny\\"

prefix=r"D:\CODE\CCONV RES\csm_mp300_50kexample_long2z+2\\"
prefix=r"D:\CODE\CCONV RES\MM-avary_csm_mp300_50kexample_long2z+2\MM-avary_csm_mp300_50kexample_long2z+2\\"
prefix=r"D:\CODE\CCONV RES\MM-x2-csm100_50kexample_long2z+2\MM-x2-csm100_50kexample_long2z+2\\"
prefix=r"D:\CODE\CCONV RES\csm_mp200_50kexample_long3\csm_mp200_50kexample_long3\\"
prefix=r"D:\CODE\CCONV RES\csm100_50kexample_long3\csm100_50kexample_long3\\"
prefix=r"D:\CODE\CCONV RES\MM-x2-pretrained_model_weights_50kexample_long2z+2\MM-x2-pretrained_model_weights_50kexample_long2z+2\\"
# prefix=r"D:\CODE\CCONV RES\pretrained_model_weights_50kexample_long3\pretrained_model_weights_50kexample_long3\\"
prefix=r"D:\CODE\CCONV RES\pretrained_model_weights_50kexample_long2z+2\\"
prefix=r"D:\CODE\CCONV RES\csm_mp300_50kmc_ball_2velx_0602\csm_mp300_50kmc_ball_2velx_0602\\"
prefix=r"D:\CODE\CCONV RES\mix\y0_csm_df300_1111emit_large\y0_csm_df300_1111emit_large\\"
prefix=r"D:/CODE/CCONV RES/mix/csm_df300_1111emit_big/csm_df300_1111emit_big/"

prefix=r"C:/Users/yzx/Downloads/emin__csm_df300_1111propeller2large21.5x/"
prefix=r"C:/Users/yzx/Downloads/emin_stable0.95__csm_df300_1111propeller2large21.5x/"
prefix=r"D:/YZX/physical/temp--DIMC_SPH/DIMC_SPH/wave_tower_output/"
prefix=r"D:/YZX/Turb Scenes/water vessel/LIZE____pretrained_model_weightswatervessel1.13x/"

# prop large
prefix=r"D:/YZX/Turb Scenes/propeller/emin__csm_df300_1111propellerlarge/emin__csm_df300_1111propellerlarge/"

# horizon
# prefix=r"D:/YZX/Turb Scenes/streamMultiObjs/LIZE___2.8Dragon__csm_df300_1111streammultiobjsHorizonDUP/"
# prefix=r"C:/Users/yzx/Downloads/LIZE_____csm300_1111streammultiobjsHorizon/"
# prefix=r"C:/Users/yzx/Downloads/emin__csm_df300_1111streammultiobjs3xcut/"


#rotatingpanel
# prefix=r"C:/Users/yzx/Downloads/emin__rot0.75csm_df300_1111rotatingpanelstatic/"
# prefix=r"D:/YZX/Turb Scenes/rotatingPanel/emin__csm_df300_1111rotatingpanelstatic/"




# prefix=r"C:/Users/yzx/Downloads/CP___emin__csm_df300_1111streammultiobjs3xcut/"

#wavetower
# prefix=r"D:\YZX\Turb Scenes\wave tower static\emin__csm_df300_1111wavetowerstatic\\"
# prefix=r"D:\YZX\Turb Scenes\wave tower static\sus320__csm_df300_1111wavetowerstatic\\"
# prefix=r"C:/Users/yzx/Downloads/emin__period0.75__csm_df300_1111wavetowerstatic/"
prefix=r"D:/YZX/Turb Scenes/wave tower static/quanti0.5____period0.75__csm_df300_1111wavetowerstatic/"
prefix=r"D:/YZX/Turb Scenes/wave tower static/quanti0.62____period0.75__csm_df300_1111wavetowerstatic/"

#prop2+large2
# prefix=r"D:/YZX/Turb Scenes/propeller/emin__csm_df300_1111propeller2+large2/"
# prefix=r"C:/Users/yzx/Downloads/cutprop____csm_df300_1111propeller2+large2/"
# prefix=r"C:/Users/yzx/Downloads/emin__rot0.5__csm_df300_1111propeller2+large2/"
# prefix=r"D:/YZX/Turb Scenes/propeller/emin__rot0.5__csm_df300_1111propeller2+large2/"
# prefix=r"C:/Users/yzx/Downloads/emin__rot0.25__csm_df300_1111propeller2+large2/"
# prefix=r"D:/YZX/Turb Scenes/propeller/emin__rot0.25__csm_df300_1111propeller2+large2/"
# prefix=r"D:/YZX/Turb Scenes/propeller/emin__rot0.25__holprop__csm_df300_1111propeller2+large2/"
# prefix=r"C:/Users/yzx/Downloads/emin__csm_df300_1111propeller2+large2fill/"
# prefix=r"C:\Users\yzx\Downloads\emin__rot.125__csm_df300_1111propeller2+large2fill\\"


#prop21.5
# prefix=r"C:/Users/yzx/Downloads/emin_sus_stable0.95__csm_df300_1111propeller2large21.5x/"
# prefix=r"C:\Users\yzx\Downloads\emin__sus150_stable0.95__0.5rot__csm_df300_1111propeller2large21.5x\\"

# prefix=r"D:\YZX\Turb Scenes\propeller\emin__csm_df300_1111propeller2large2\\"



# prefix=r"D:/YZX/Turb Scenes/streamMultiObjs/emin__csm_df300_1111streammultiobjs3xcutbunny12/"

# prefix=r"C:/Users/yzx/Downloads/emin__csm_df300_1111streammultiobjs3xcut/"

# prefix=r"D:\YZX\Turb Scenes\streamMultiObjs\emin__csm_df300_1111streammultiobjs3xcuthorizon\\"

#ds
# prefix=r"D:\YZX\physical\zzcsm-302-351\csm_304_output\\"
# prefix=r"C:/Users/yzx/Documents/CODE/temp-MCVSPH-FORK/csm_mt_v2_304_output/"
# prefix=r"C:/Users/yzx/Documents/CODE/temp-MCVSPH-FORK/sp_mp_0602__300_output/"

# prefix=r"C:\Users\yzx\Downloads\\"
prefix=r"C:/Users/yzx/Downloads/temp/"


filepre=r"fluid_"
# filepre=r"particle_object_0_"
prm_formatnum=1
prm_savevel=0



acurlabs=[]
acurlvar=[]


# prm
lv=0
rv=3600
# rv=lv+1
prm_single=0

prm_step=1
# prm_step=10
# prm_step=50

dt_frame=0.016
prm_exportRGB=0

prmcalcrigidcenter=0

prm_exportCurl=0
prmvnormxoz=0

prmexportang=0
prmexportidx=0
prmyoz=0

prm_colorcase=1
prm_loadply=1

ti.init(arch=ti.gpu,
         device_memory_fraction=0.9,
         debug=False,
         random_seed=int(1234),kernel_profiler=False)




# data = np.load(prefix+r"fluid_0955.npz")
# 通过键名获取数据
# avel=[]
# apos=[]
# data=0

# avel = data['vel']
# apos = data['pos']
# print(avel.shape)

particlenum=0
particle_max_num=540000
particle_max_num=660000
# particle_max_num=1030000
# particle_max_num=2200000






particle_radius=0.025
support_radius = particle_radius * 4.0 
particle_diameter = 2 * particle_radius
m_V0 = 0.8 * particle_diameter ** 3



# m_V     = ti.field(dtype=float, shape=particle_max_num)
# x = ti.Vector.field(3, dtype=float, shape=particle_max_num)
# tempa=ti.Vector.field(3, dtype=float, shape=particle_max_num)
# v = ti.Vector.field(3, dtype=float, shape=particle_max_num)
# particle_color = ti.Vector.field(3,dtype=float,shape=particle_max_num)


# m_V.fill(m_V0)

# x.from_numpy(apos)
# v.from_numpy(avel)

# rec=x.to_numpy()

# print(rec.shape)
# print(apos[123])
# print(x[123])
# print(rec[123])







@ti.data_oriented
class curlvisual:

    def __init__(self) -> None:
        self.curlabs     = ti.field(dtype=float, shape=particle_max_num)
        self.x = ti.Vector.field(3, dtype=float, shape=particle_max_num)
        self.v = ti.Vector.field(3, dtype=float, shape=particle_max_num)

    @ti.func
    def cubic_kernel_derivative(self, r):
        h = support_radius
        # derivative of cubic spline smoothing kernel
        

        k = 8 / np.pi
        k = 6. * k / h ** 3
        r_norm = r.norm()
        q = r_norm / h
        res = ti.Vector([0.0 for _ in range(3)])
        if r_norm > 1e-5 and q <= 1.0:
            grad_q = r / (r_norm * h)
            if q <= 0.5:
                res = k * q * (3.0 * q - 2.0) * grad_q
            else:
                factor = 1.0 - q
                res = k * (-factor * factor) * grad_q
        return res




    @ti.func
    def compute_particles_color_curl_task(self, p_i, p_j,
                                            curl_v: ti.template()):

        x_i = self.x[p_i]
        x_j = self.x[p_j]
        
        #zxc 涡度计算公式
        curl_v += m_V0 * (
            self.v[p_j] - self.v[p_i]).cross(
                self.cubic_kernel_derivative(x_i - x_j))


    #zxc 原来的实现中加入了背景网格。并且每模拟一步，都会重新对粒子进行编号。这里不需要。
    @ti.func
    def for_all_neighbors(self, p_i, task: ti.template(), ret: ti.template()):

        for p_j in range(0,particlenum):
            if p_i[0] != p_j and (self.x[p_i] - self.x[p_j]).norm() < support_radius:
                task(p_i, p_j, ret)
    # @ti.func
    # def curl_color(self, v: ti.template(), w: ti.template()):
    #     v_norm = v.norm()


    #     if  (prm_colorcase==0):
    #         w[0] = -ti.exp(-0.03 * v_norm) + 1
    #     elif(prm_colorcase==1):
    #         w[0] = -ti.exp(-0.06 * v_norm) + 1

    #     w[1] = w[0]

    @ti.kernel
    def compute_particles_color_curl(self):#1
        for p_i in ti.grouped(self.x):
            # print(p_i)
            if(p_i[0]>particlenum):
            #     # print(p_i[0])
            #     # print('cont')
            #     ti.atomic_add(self.conttime[0],1)
                continue
            # tempa[p_i]=ti.Vector([6,6,6])
            
            #if self.ps.is_in_dynamic_area[p_i] == True:
            #color_base = ti.Vector([0.196,0.392,0.784])
            # color_base = ti.Vector([0.0, 0.0, 1.0])
            # color_vis_curl = ti.Vector([0.0, 0.0, 0.0])
            v_curl = ti.Vector([0.0, 0.0, 0.0])

            #计算结束后结果送入v_curl
            self.for_all_neighbors(
                p_i, self.compute_particles_color_curl_task, v_curl)
            # self.ps.vorticity_eva[p_i] = v_curl
            if(prm_exportCurl):
                self.curlabs[p_i]=v_curl.norm()
            # self.curl_color(v_curl, color_vis_curl)
            # particle_color[p_i] = ti.math.clamp(
            #     color_base + color_vis_curl, 0.1, 1.0)
          
obj1=curlvisual()
#COPY
def loadply(filename,idx):

    global particlenum
    
    if(not prm_formatnum):
        fn0=     filename+str(idx)+'.ply'
        if(not prm_savevel):
            fn1=filename+str(idx+1)+'.ply'
    else:
        fn0=    filename+str('{0:04d}'.format(idx))+'.ply'
        if(not prm_savevel):
            fn1=filename+str('{0:04d}'.format(idx+1))+'.ply'


    pos0=get1ply(fn0).astype('float32')

    if(prmcalcrigidcenter):
        return pos0

    if(not prm_savevel):
        pos1=get1ply(fn1)[:pos0.shape[0],:].astype('float32')




    particlenum=pos0.shape[0]




    obj1.x.from_numpy(pos0)
    # print(pos0.shape)
    if(prm_savevel):
        obj1.v.from_numpy(np.load(filename+str('{0:04d}'.format(idx))+'.npz')['vel'])
    else:
        obj1.v.from_numpy((pos1-pos0)/dt_frame)
    



# def loadnpz(filename):

#     global avel,apos
#     global particlenum
#     global x,v

#     data = np.load(filename)
#     avel = data['vel']
#     apos = data['pos']
#     # print('[apos]')
#     # print(apos.shape)



#     particlenum=apos.shape[0]
    
#     x.from_numpy(apos)
#     v.from_numpy(avel)




#swi

if(prm_single):
    if __name__ == "__main__":
        parser = argparse.ArgumentParser(description='xx')
        parser.add_argument('--lv',
                            default='1',
                            help='scene file')
        parser.add_argument('--rv',
                            default='1',
                            help='scene name')
        parser.add_argument('--emitend',
                            default='1',
                            help='scene name')

        args = parser.parse_args()
        lv = int(args.lv)
        rv = int(args.rv)

        print('lv from bat')
        if(lv==rv):
            print('[SINGLE FRAME]')



for i in tqdm(range(lv,rv+1,prm_step)):
    # obj1.curlabs.fill(0)
    # obj1.x.fill(0)
    # obj1.v.fill(0)
    # assert(np.allclose(np.zeros_like(obj1.curlabs.to_numpy()),obj1.curlabs.to_numpy()))
    # assert(np.allclose(np.zeros_like(obj1.x.to_numpy()),obj1.x.to_numpy()))
    # assert(np.allclose(np.zeros_like(obj1.v.to_numpy()),obj1.v.to_numpy()))

    if(prmcalcrigidcenter):
        print('center')
        # pos0=loadply(prefix+"rigid_",i)

        # #temp
        # pos0=loadply(prefix+"fluid_",i)
        # pos1=loadply(prefix+"fluid_",i)
        # dy=(pos1-pos0)[:,1]
        # v0=np.mean((dy+(9.81*0.016*0.016/2))/0.016)
        # print(v0)
        # assert(False)


        mask=pos0[:,0]>8
        print(pos0.shape)
        print(np.sum(mask.astype(np.int16)))
        cx=np.mean(pos0[mask,0])
        cy=np.mean(pos0[mask,1])
        cz=np.mean(pos0[mask,2])
        print(cx)
        print(cy)
        print(cz)

    
        position = np.array([cx,cy,cz])
 
        # 计算每个点与给定位置之间的欧几里得距离
        distances = np.linalg.norm(pos0 - position, axis=1)
 
        # 找出距离小于 0.001 的点的索引
        indices = np.where(distances < 0.07)[0]
        
        print("Indices of points within distance 0.001:", indices)

        assert(False)
    if(prm_loadply):
        loadply(prefix+filepre,i)



                 
               


    else:
        loadnpz((prefix+filepre+'{0:04d}'+r".npz").format(i))
    # temp=x.to_numpy()


 
    # print('[value]')
    # print(particlenum)
    # print(temp[120])
    # print(temp[particlenum-1])
    # print(temp[particlenum])



    
    obj1.compute_particles_color_curl()
    # assert(np.allclose(np.zeros_like(obj1.curlabs.to_numpy()[particlenum:]),obj1.curlabs.to_numpy()[particlenum:]))
    # assert(np.allclose(np.zeros_like(obj1.x.to_numpy()[particlenum:]),obj1.x.to_numpy()[particlenum:]))
    # assert(np.allclose(np.zeros_like(obj1.v.to_numpy()[particlenum:]),obj1.v.to_numpy()[particlenum:]))
   




    if(prm_exportCurl):
        curlabsn         =obj1.curlabs.to_numpy()[0:particlenum]
        # acurlabs.append(np.sum(curlabsn)/particlenum)
        # acurlvar.append(np.var(curlabsn,ddof=1))
        if(prm_loadply):
            if(prm_formatnum):
                f_write=(prefix+filepre+'{0:04d}'+r".ply").format(i)
            else:
                f_write=prefix+filepre+str(i)+r".ply"
            print(curlabsn.shape)

            plyaddattr(f_write,
                curlabsn,
                'curlabs')
            
        else:
            pass
            plyaddattr((prefix+filepre+'{0:04d}'+r".ply").format(i),
                    curlabsn,
                    'curlabs')
            
    elif(prmexportang):
        v        =obj1.v.to_numpy()[0:particlenum]
   
        if(prmyoz):
            angle = np.arctan2(v[:,2], v[:,1])  # 计算角度，范围 [-π, π]
        else:
            angle = np.arctan2(v[:,2], v[:,0])
     

        angle[angle < 0] += 2 * np.pi  # 调整到 [0, 2π]
        # print(angle)
        
        if(prmexportidx):
            angle=np.arange(1,angle.shape[0]+1)

        # print(ang.shape)
        # assert(False)
        f_write=(prefix+filepre+'{0:04d}'+r".ply").format(i)
        
     

        plyaddattr(f_write,
                angle,
                'ang')
        

    else:
        vnorm         =obj1.v.to_numpy()[0:particlenum]
        vnorm=np.sqrt(np.sum(vnorm**2,axis=1))
        if(prmvnormxoz):
            # print('XOZ VNORM')
            vnorm         =obj1.v.to_numpy()[0:particlenum,[0,2]]
            vnorm=np.sqrt(np.sum(vnorm**2,axis=1))

        if(prm_formatnum):
            f_write=(prefix+filepre+'{0:04d}'+r".ply").format(i)
        else:
            f_write=(prefix+filepre+str(i)+r".ply")

        plyaddattr(f_write,
                vnorm,
                'vnorm')
    # particle_colorn=particle_color.to_numpy()[0:particlenum]
    # print(particle_colorn.shape)
    # print('[zxc]')
    # print(particlenum)
    # print(curlabsn.shape)
    # print(particle_colorn.shape)
    
    # plyaddattr((prefix+filepre+'{0:04d}'+r".ply").format(i),
    #         particle_colorn[:,0],
    #         'red')
    # if(prm_exportRGB):
    #     plyaddRGB((prefix+filepre+'{0:04d}'+r".ply").format(i),
    #             particle_colorn)
    

    if(i%20==0):
        print(str(i)+' done')



#prm
# np.save(prefix+'av_curl_',acurlabs)
# np.save(prefix+'av_curl_var_',acurlvar)
