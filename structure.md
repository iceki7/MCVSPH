env
    taichi  1.7   
    python  3.9

self.delta_vorticity
self.ps.v
self.ps.x[p_i]
self.ps.acceleration[p_i]



self.ps.material[p_j] == self.ps.material_fluid:

#prm
record_endtime

VS code python cmd
C:/Users/123/.conda/envs/ev1/python.exe .\run_simulation.py --scene_file .\data\scenes\
C:/Users/123/.conda/envs/ev1/python.exe .\run_simulation.py --scene_file .\randomScene\


【数据内容】
lowfluid        0   120     700         每隔4个step输出一帧。
                0   30      175




【训练速度】
lowfluid-r  总计240 seg     0.4 ips
our_default_data    0.77 ips
lowfluid6           1.55 ips


训练速度应该和数据量没关系。这里的数据就是循环读入。它不会增加网络梯队回传的时间。
（但是读文件的时间增多了，原先读过一次的文件就会保存下来，这里是存为dict。但即便如此也只有第一次读入文件时需要读，之后都是读dict）
是不是dict的查找效率太低？因为要反复查。总共也只存几次，之后都是大量的查找。

是不是转换成tensor的效率太低?
    转成tensor了，不行


【场景特征】

    其实碰撞有点激烈。是否要降低一下下落高度


ball

湍的细节在流体冲击sphere前，以及到达边界反弹后。
侧面可以看到涡旋
以及交替往返的波浪



【data augment】

 随机旋转。
        1.每一帧随机旋转 但是那样每一帧都要加一个盒子

        2.每个场景随机旋转。这样只要R乘上初始参数就行了
            不行。初始参数要途径MCVSPH变成水块。
            要么实现一个放置倾斜水块的方法，要么就生成出来再倾斜。
            而且不能逐帧倾斜，太麻烦了

        R*box
        R*fluid