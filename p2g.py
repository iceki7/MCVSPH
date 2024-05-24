
# velocityRasterize
import taichi as ti


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

#prm_
#gridsize=0.4   equalaxis=0

gridsize=0.1
equalaxis=1
particlemaxnum=50000
dim=3
domain_start=np.array([-1,  0,      -4.7])
domain_end=  np.array([1,   2.6,    4])


domain_size=domain_end-domain_start

gridnumx=int(domain_size[0]/gridsize)+1
gridnumy=int(domain_size[1]/gridsize)+1

gridnum=gridnumx*gridnumy
if(dim==3):
    gridnumz=int(domain_size[2]/gridsize)+1
    gridnum*=gridnumz

#邻域半径
support_radius=gridsize
particlenum=0

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

    print('[coord shape]')
    print(coord.shape)
    # print(coord);exit(0)
    return coord

gridpos=  ti.Vector.field(dim, dtype=float, shape=gridnum)
gridvel=  ti.Vector.field(dim, dtype=float, shape=gridnum)
partvel=  ti.Vector.field(dim, dtype=float, shape=particlemaxnum)
partpos=  ti.Vector.field(dim, dtype=float, shape=particlemaxnum)

gridpos.from_numpy(initGridCoord())

@ti.data_oriented
class velp2g:

    @ti.func
    def for_all_neighborsPG(self, g_i, task: ti.template(), ret: ti.template()):

        for p_j in range(0,particlenum):
            if (gridpos[g_i] - partpos[p_j]).norm() < support_radius:
                # print('[pair]')
                # print(g_i)
                # print(p_j)
                task(g_i, p_j, ret)

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
    def velInterpolate_task(self, g_i, p_j,
                                            curl_v: ti.template()):
        gridvel[g_i]+=partvel[p_j]#*\

            # (gridpos[g_i] - partpos[p_j]).norm()
            # self.cubic_kernel_derivative(gridpos[g_i] - partpos[p_j])


   
    @ti.kernel
    def velInterpolate(self):
        for g_i in ti.grouped(gridpos):
            v_curl = ti.Vector([0.0, 0.0, 0.0])
            gridvel[g_i]=v_curl
            self.for_all_neighborsPG(
                g_i, self.velInterpolate_task, v_curl)
            # print(gridvel[g_i])
            


def loadfile(filename):

    global particlenum

    data = np.load(filename)

    
    partpos.from_numpy(data['pos'])
    partvel.from_numpy(data['vel'])
    particlenum=data['pos'].shape[0]
    # print(particlenum);exit(0)
    # print(partvel);exit(0)

#prm 
lv=1
rv=1000

cnt=0
prefix=r"D:\CODE\CCONV RES\MM-x2-pretrained_model_weights_50kexample_long2z+2\MM-x2-pretrained_model_weights_50kexample_long2z+2\\"
filepre=r"fluid_"
def quiver(x,y,z,vel):
    import matplotlib.pyplot as plt
    import numpy as np

    ax = plt.figure().add_subplot(projection='3d')

    # print('[quiver]')
    # print(x.shape)
    # print(vel.shape)


    #REF     https://matplotlib.org/3.4.2/gallery/mplot3d/quiver3d.html
    ax.quiver(x, y, z, np.reshape(vel[:,0],x.shape),\
                       np.reshape(vel[:,1],x.shape),\
                       np.reshape(vel[:,2],x.shape),\
                         length=0.1, normalize=True)


    ax.set_xlim3d(domain_start[0],domain_end[0])
    ax.set_ylim3d(domain_start[1],domain_end[1])
    ax.set_zlim3d(domain_start[2],8)

    global cnt

    
    if(equalaxis):
        ax.set_aspect('equal')
    plt.savefig('temp'+str(cnt)+'.png',bbox_inches='tight',pad_inches=0.0, dpi=300)

    cnt+=1
    # plt.show()

for i in range(lv,rv+1,10):
    loadfile((prefix+filepre+'{0:04d}'+r".npz").format(i))
    print(i)
    obj1=velp2g()
    obj1.velInterpolate()
    gridvelnp=gridvel.to_numpy()
    # print(gridvelnp)
    quiver(X,Y,Z,gridvelnp)
    # print(gridvel.shape)


