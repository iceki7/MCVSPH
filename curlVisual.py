import numpy as np
import taichi as ti

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


filepre=r"fluid_"
# filepre=r"particle_object_0_"
prm_formatnum=1
prm_savevel=0



acurlabs=[]
acurlvar=[]


lv=0
rv=999
prm_step=1

dt_frame=0.016
prm_exportRGB=0
prm_exportCurl=1
prm_colorcase=1
prm_loadply=1

ti.init(arch=ti.gpu,
         device_memory_fraction=0.9,
         debug=False,
         random_seed=int(1234),kernel_profiler=False)


# prefix=r"C:\Users\123\Downloads\lowfluidS100_50kexample_scene\\"

# data = np.load(prefix+r"fluid_0955.npz")
# 通过键名获取数据
avel=[]
apos=[]
data=0

# avel = data['vel']
# apos = data['pos']
# print(avel.shape)

particlenum=0
particle_max_num=150000




particle_radius=0.025
support_radius = particle_radius * 4.0 
particle_diameter = 2 * particle_radius
m_V0 = 0.8 * particle_diameter ** 3



m_V     = ti.field(dtype=float, shape=particle_max_num)
x = ti.Vector.field(3, dtype=float, shape=particle_max_num)
v = ti.Vector.field(3, dtype=float, shape=particle_max_num)
particle_color = ti.Vector.field(3,dtype=float,shape=particle_max_num)


m_V.fill(m_V0)

# x.from_numpy(apos)
# v.from_numpy(avel)

# rec=x.to_numpy()

# print(rec.shape)
# print(apos[123])
# print(x[123])
# print(rec[123])


import taichi as ti



#COPY
def loadply(filename,idx):

    global x,v,particlenum
    
    if(not prm_formatnum):
        fn= filename+str(idx)+'.ply'
        if(not prm_savevel):
            fn1=filename+str(idx+1)+'.ply'
    else:
        fn= filename+str('{0:04d}'.format(idx))+'.ply'
        if(not prm_savevel):
            fn1=filename+str('{0:04d}'.format(idx+1))+'.ply'

    pos0=get1ply(fn)
    if(not prm_savevel):
        pos1=get1ply(fn1)[:0,pos0.shape[0]]




    particlenum=pos0.shape[0]
    x.from_numpy(pos0)
    if(prm_savevel):
        v.from_numpy(np.load(filename+str('{0:04d}'.format(idx))+'.npz')['vel'])
    else:
        v.from_numpy((pos1-pos0)/dt_frame)


def loadnpz(filename):

    global avel,apos
    global particlenum
    global x,v

    data = np.load(filename)
    avel = data['vel']
    apos = data['pos']
    # print('[apos]')
    # print(apos.shape)



    particlenum=apos.shape[0]
    
    x.from_numpy(apos)
    v.from_numpy(avel)

@ti.data_oriented
class curlvisual:

    def __init__(self) -> None:
        self.curlabs     = ti.field(dtype=float, shape=particle_max_num)

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
        x_i = x[p_i]
        x_j = x[p_j]
        
        #zxc 涡度计算公式
        curl_v += m_V[p_j] * (
            v[p_j] - v[p_i]).cross(
                self.cubic_kernel_derivative(x_i - x_j))


    #zxc 原来的实现中加入了背景网格。并且每模拟一步，都会重新对粒子进行编号。这里不需要。
    @ti.func
    def for_all_neighbors(self, p_i, task: ti.template(), ret: ti.template()):

        for p_j in range(0,particlenum):
            if p_i[0] != p_j and (x[p_i] - x[p_j]).norm() < support_radius:
                task(p_i, p_j, ret)
    @ti.func
    def curl_color(self, v: ti.template(), w: ti.template()):
        v_norm = v.norm()


        if  (prm_colorcase==0):
            w[0] = -ti.exp(-0.03 * v_norm) + 1
        elif(prm_colorcase==1):
            w[0] = -ti.exp(-0.06 * v_norm) + 1

        w[1] = w[0]

    @ti.kernel
    def compute_particles_color_curl(self):#1
        for p_i in ti.grouped(x):
            #if self.ps.is_in_dynamic_area[p_i] == True:
            #color_base = ti.Vector([0.196,0.392,0.784])
            color_base = ti.Vector([0.0, 0.0, 1.0])
            color_vis_curl = ti.Vector([0.0, 0.0, 0.0])
            v_curl = ti.Vector([0.0, 0.0, 0.0])

            #计算结束后结果送入v_curl
            self.for_all_neighbors(
                p_i, self.compute_particles_color_curl_task, v_curl)
            # self.ps.vorticity_eva[p_i] = v_curl
            if(prm_exportCurl):
                self.curlabs[p_i]=v_curl.norm()
            self.curl_color(v_curl, color_vis_curl)
            particle_color[p_i] = ti.math.clamp(
                color_base + color_vis_curl, 0.1, 1.0)
          
obj1=curlvisual()



for i in range(lv,rv,prm_step):
    if(prm_loadply):
        loadply(prefix+filepre,i)

    else:
        loadnpz((prefix+filepre+'{0:04d}'+r".npz").format(i))
    


    obj1.compute_particles_color_curl()
    if(prm_exportCurl):
        curlabsn         =obj1.curlabs.to_numpy()[0:particlenum]
        acurlabs.append(np.sum(curlabsn)/particlenum)
        acurlvar.append(np.var(curlabsn,ddof=1))
        if(prm_loadply):
            if(prm_formatnum):
                fn=(prefix+filepre+'{0:04d}'+r".ply").format(i)
            else:
                fn=prefix+filepre+str(i)+r".ply"
            # print(curlabsn.shape)

            plyaddattr(fn,
                curlabsn,
                'curlabs')
            
        else:
            plyaddattr((prefix+filepre+'{0:04d}'+r".ply").format(i),
                    curlabsn,
                    'curlabs')
    particle_colorn=particle_color.to_numpy()[0:particlenum]
    # print(particle_colorn.shape)
    # print('[zxc]')
    # print(particlenum)
    # print(curlabsn.shape)
    # print(particle_colorn.shape)
    
    # plyaddattr((prefix+filepre+'{0:04d}'+r".ply").format(i),
    #         particle_colorn[:,0],
    #         'red')
    if(prm_exportRGB):
        plyaddRGB((prefix+filepre+'{0:04d}'+r".ply").format(i),
                particle_colorn)
    

    if(i%20==0):
        print(str(i)+' done')
print('[all done]')
#prm
np.save(prefix+'av_curl_',acurlabs)
np.save(prefix+'av_curl_var_',acurlvar)
