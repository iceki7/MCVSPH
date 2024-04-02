import numpy as np


def zxcboxandnorm(lb,rt,rad):#know

    boxn=[]
    boxp=[]

    boxsize=rt-lb
    partnumx=   int((boxsize[0]/(2*rad)))+1
    partnumy=   int((boxsize[1]/(2*rad)))+1
    partnumz=   int((boxsize[2]/(2*rad)))+1
    # print(partnumx)
    # print(partnumy)
    # print(partnumz)
    for i in (range(partnumx)):
        for j in range(partnumy):
            for k in range(partnumz):
                if(    i!=0 and i!=partnumx-1 \
                   and j!=0 and j!=partnumy-1 \
                   and k!=0 and k!=partnumz-1):#internal
                    # print(str(i+1)+","+str(j+1)+","+str(k+1))
                    continue

                if(i==0):
                    boxn.append([1.,0,0])
                elif(i==partnumx-1):
                    boxn.append([-1.,0,0])

                elif(j==0):
                    boxn.append([0,1.,0])
                elif(j==partnumy-1):
                    boxn.append([0,-1.,0])

                elif(k==0):
                    boxn.append([0,0,1.])
                elif(k==partnumz-1):
                    boxn.append([0,0,-1.])



                boxp.append([lb[0]+i*rad*2,
                             lb[1]+j*rad*2,
                             lb[2]+k*rad*2])
                
    boxp=np.array(boxp)
    boxn=np.array(boxn)

    boxp.astype(np.float32)
    boxn.astype(np.float32)

    # print('[box]\t'+str(boxp.shape))
    # print(boxp.dtype)

    return boxp,boxn

