import numpy as np
import taichi as ti
import tqdm
import open3d as o3d        #python3.8







ti.init(arch=ti.gpu,
         device_memory_fraction=0.9,
         debug=False,
         random_seed=int(1234),kernel_profiler=False)




particlenum=0
particle_max_num=150000




vel = ti.Vector.field(3, dtype=float, shape=particle_max_num)
vor = ti.Vector.field(3, dtype=float, shape=particle_max_num)



def loadply(filename,idx):

    global vel,particlenum

    # velnp=get1ply(filename+r"velocity_object_0_"+str(idx)+".ply")
    # vornp=get1ply(filename+r"vorticity_object_0_"+str(idx)+".ply")

   

    pcd=o3d.io.read_point_cloud(filename+r"velocity_object_0_"+str(idx)+".ply")
    velnp=np.asarray(pcd.points)
    # print(velnp.shape)
    # exit(0)

    pcd=o3d.io.read_point_cloud(filename+r"vorticity_object_0_"+str(idx)+".ply")
    vornp=np.asarray(pcd.points)


    particlenum=velnp.shape[0]

    vel.from_numpy(velnp)
    vor.from_numpy(vornp)




@ti.data_oriented
class curlvisual:

    def __init__(self) -> None:
        self.vel_abs     = ti.field(dtype=float, shape=particle_max_num)
        self.vor_abs     = ti.field(dtype=float, shape=particle_max_num)
        self.avvel=ti.field(dtype=float, shape=(1,1))
        self.avvor=0
 
   

    @ti.kernel
    def compute_particles_color_curl(self):#1
        for i in ti.grouped(vel):
            temp=vel[i].norm()
            # self.avvel+=temp
            # ti.atomic_add(self.avvel, temp)
            self.vel_abs[i]=temp


            temp=vor[i].norm()
            # self.avvor+=temp
            self.vor_abs[i]=temp


         
          
obj1=curlvisual()

# prm_
lv=0
rv=3999
prefix=r"F:\yzx-2\MCVSPH-FORK\specific_df_1_output\\"


a_avvel=[]
a_avvor=[]

for i in tqdm.tqdm(range(lv,rv+1)):

    loadply(prefix,i)


    obj1.compute_particles_color_curl()

    if(i%20==0):
        print(i)

    avvel=obj1.vel_abs.to_numpy()[0:particlenum]
    avvor=obj1.vor_abs.to_numpy()[0:particlenum]

    # print(avvel.shape)#partnum 1
    avvel=np.mean(avvel)
    avvor=np.mean(avvor)


    a_avvel.append(avvel)
    a_avvor.append(avvor)


    

np.save(prefix+"av_vel",a_avvel)
np.save(prefix+"av_vor",a_avvor)
