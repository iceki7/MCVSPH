import os
from PIL import Image,ImageEnhance 
from matplotlib import pyplot as plt
import numpy as np
import scipy.fft
import scipy.fftpack as fp      #legacy
import scipy
from tqdm import tqdm

import writeVTK
np.random.seed(1234)

# https://stackoverflow.com/questions/38476359/fft-on-image-with-python     answer1


#PARAM DEFAULT
prm_flt=0
trans_case='B'
flt_case='B'
dim=3
useratio=0
prm_vecnum=3#速度分量个数

# NOT PARAM
threshigh=0
threslow=0
threshigh_ratio=0
threslow_ratio=0
hasflt=0
flt=None




## Functions to go from image to frequency-image and back
im2freq = lambda data: fp.rfft(fp.rfft(data, axis=0),
                               axis=1)
#迭代FFT，先对一个轴做，然后再对另一个轴做



freq2im = lambda f: fp.irfft(fp.irfft(f, axis=1),
                             axis=0)


def visualRes(ori,idx):
    print('[img]'+str(np.mean(np.imag(ori))))
    ori=np.real(ori)
    print(ori.shape)
    print(np.mean(ori))
    print('[vel final]')


    if(prm_vecnum==3):
        temppos=np.load(predir+r"gridpos.npy")
        if(prm_flt):
            fname=predir+"vel-fft-thre"+str(threslow)+'t'+str(threshigh)+'--'+str(idx)
        else:
            fname=predir+"vel-"+str(idx)

        writeVTK.writeVTK(vertices=temppos,\
                    velocities=ori,\
                fname=fname,)
        
    if(prm_vecnum==1):
        np.save(predir+"vel-fft-mat-"+str(idx),ori)
        plt.imshow(ori[slicey,:,:].T, cmap='hot', \
                #    interpolation='none'\
                    # interpolation='gaussian',\
                    # extent=(domain_start[0],domain_end[0],\
                    #         domain_start[2],domain_end[2])  ,
                    vmin=0,
                    vmax=13000
                    )#know
        plt.colorbar()
        plt.savefig(predir+'vel-fft-'+str(idx)+'slice'+str(slicey)+'-thre'+str(threslow)+'t'+str(threshigh)+'.png',\
                    bbox_inches='tight',pad_inches=0.0, dpi=300)
        plt.close()


def AmplitudeA(freq):
    x=[]
    y=[]
    global predir
    for index, value in np.ndenumerate(freq):
        r2=index[0]**2+index[1]**2
        x.append(r2)
        y.append(value)
    plt.xlim((0,10000))
    plt.scatter(x,y)

    plt.savefig(predir+'[amp].png')
    plt.close()
    
# https://blog.csdn.net/u011555996/article/details/104264500
def AmplitudeB(freq,idx):
    
    temp=predir+"amp-"+str(idx)+'.npy'

    #如果之前计算过频谱，就不重新算了，直接绘图
    if( os.path.exists(temp)):
        print('[use amp cache]')
        tempy=np.load(temp)
        tempx=np.arange(len(tempy))
    else:
        print('[calc amp]')
       
        plotdict={}
        pixelnum={}#每个freq对应的pixelnum
        centerx=freq.shape[0]/2
        centery=freq.shape[1]/2

        if(dim==3):
            centerz=freq.shape[2]/2


        for index, value in np.ndenumerate(freq):
            if(dim==2):
                dis=abs(index[0]-centerx)+abs(index[1]-centery)
            elif(dim==3):
                dis=abs(index[0]-centerx)+\
                    abs(index[1]-centery)+\
                    abs(index[2]-centerz)

            dis=int(dis)
            
            if dis in plotdict:
                plotdict[dis]+=value
                pixelnum[dis]+=1
            else:
                plotdict[dis]=value
                pixelnum[dis]=1
            # print('index\t'+str(index))



        tempx=np.array(list(plotdict.keys()))
        tempy=np.array(list(plotdict.values()))
        np.save(predir+"amp-"+str(idx),tempy)
        print(freq.shape)
        print('freq num'+str(len(plotdict)))      
        print(tempy.shape)
        print('[freq strength]')
        print(np.max(np.real(tempy)))
        print(np.mean(np.real(tempy)))
        print(np.min(np.real(tempy)))




    #可视化调整
    # tempy[np.absolute(tempy)<1e7]=0
    from scipy.ndimage import gaussian_filter1d 
    tempy=gaussian_filter1d(tempy,sigma=1.5)
    # tempy=gaussian_filter1d(tempy,sigma=3)
    # plt.yscale("log")
    # plt.plot(tempx,tempy/tempn)
    # plt.ylim(-1.5e8,1.5e8)
    # plt.xlim((0,10000))
    # plt.ylim((0,10000))
    # plt.yscale("log")
    # plt.yscale("ex")
    plt.xlabel("freq")
    plt.ylabel("strength")


    plt.plot(tempx,tempy)
    plt.savefig(predir+'[amp]-'+str(idx)+'.png')
    plt.close()
    print('drawing done')

    


#trans_case=B 的频谱图是以原点为中心对称的，所以flt也是先保留中间区域的
def makefltB(freq):

    centerx=freq.shape[0]/2
    centery=freq.shape[1]/2
    if(dim==3):
        centerz=freq.shape[2]/2

    print('[making filterB]')


    global flt
    flt=np.zeros_like(freq)

    for index, value in np.ndenumerate(flt):#know
        if(dim==2):
            dis=abs(index[0]-centerx)+\
                abs(index[1]-centery)
            
            dismax=(freq.shape[0]+freq.shape[1])/2
            if(useratio):
                if(    dis/dismax<threshigh_ratio \
                and dis/dismax>threslow_ratio ):
                    flt[index]=1
            else:    
                if(    dis<threshigh \
                and dis>threslow ):
                    flt[index]=1
            
            

        elif(dim==3):
            dis=abs(index[0]-centerx)+\
                abs(index[1]-centery)+\
                abs(index[2]-centerz)
            dismax=(freq.shape[0]+freq.shape[1]+freq.shape[2])/2
            
            if(useratio):
                if(    dis/dismax<threshigh_ratio \
                and dis/dismax>threslow_ratio ):
                    flt[index]=1 
            else:
                if(     dis < threshigh \
                    and dis > threslow ):
                    flt[index]=1 

    if(useratio):
        pass
    else:
        print('cut freq:'+str(threslow)+'-'+str(threshigh))    
    print(np.sum(flt))
    print(flt.shape[0]*flt.shape[1])
    
    
    # return int(threslow*dismax),\
    #         int(threshigh*dismax)
    return threslow,threshigh

def makefltA(freq):
    #以扇形区域删减频谱图。适用于caseA.max thers:1.42
 
    print('[making filter]')
    global flt
    flt=np.zeros_like(freq)
    
    for index, value in np.ndenumerate(flt):#know
        # print(index)
        # print(index[0])
        r2=index[0]**2+index[1]**2
        if(    r2 <=   (threshigh_ratio*freq.shape[0])**2\
           and r2 >=   (threslow_ratio*freq.shape[0])**2 ):
            flt[index]=1

    print(np.sum(flt))
    print(flt.shape[0]*flt.shape[1])
    print('[filter done]')



## Helper functions to rescale a frequency-image to [0, 255] and save




def myfft(data,idx):
    


    if(dim==2):
        #2D method
        if(trans_case=='A'):
            freq=fp.rfft(fp.rfft(data, axis=0),axis=1)
        elif(trans_case=='B'):
            freq=scipy.fft.fft2(data)#shape gridnum gridnum    
        



        # freq=fp.fft2(data)
        # freq=scipy.fft.rfft2(data)        #输出的尺寸不对

    elif(dim==3):
        #3D method
        # print(np.sum(np.isnan(data)))
        # print(np.max(data))
        # print(np.min(data))
        # print(np.mean(data))
        # print(data.dtype)
        # print(data[0,0,0])
        if(trans_case=='A'):
            assert(False)
        elif(trans_case=='B'):
            data=data.astype('float64')
            freq=scipy.fft.fftn(data)

      
    if(trans_case=='B'):

        freqs=scipy.fft.fftshift(freq)
        assert(freqs.shape==freq.shape)
    else:
        freqs=freq
       
        

 
    if(dim==2):
 
        temp=(freqs-np.min(freqs)) / (np.max(freqs)-np.min(freqs))
        temp*=256
        temp=temp.astype('uint8')
        print(temp.shape)
        temp=np.tile(temp,(3,1,1))#know
        temp=np.swapaxes(temp,0,1)
        temp=np.swapaxes(temp,1,2)
        print(temp.shape)#L W 3
        img=Image.fromarray(temp, 'RGB')


        #以下为增强视效的处理
        # enhancer = ImageEnhance.Contrast(img)  
        # img_enhanced = enhancer.enhance(factor=2.0)  
        # img_enhanced.save(predir+r'-freq.jpg')  


        img.save(predir+r'-freq.jpg')


    if(trans_case=='B' and prm_vecnum==1):
        AmplitudeB(freqs,idx=idx)



    if(prm_flt):
        global flt,hasflt
        if(not hasflt):
            if(flt_case=='A'):
                makefltA(freqs)
            elif(flt_case=='B'):
                makefltB(freqs)

            hasflt=1
        freqs*=flt


    if(trans_case=='B'):
        freq=scipy.fft.ifftshift(freqs)
    else:
        freq=freqs


    if(dim==2):
        if(trans_case=='A'):
            ori=fp.irfft(fp.irfft(freq,axis=1),axis=0)
        elif(trans_case=='B'):
            ori=scipy.fft.ifft2(freq)   
        # ori=fp.ifft2(freq)

        print(ori.shape)
        

        ori=np.repeat(ori[np.newaxis, ...], 3, axis=0)   #know
        ori=np.swapaxes(ori,0,1)
        ori=np.swapaxes(ori,1,2)

        # out.putdata(map(tuple, ori.reshape(-1, 3)))
        img=Image.fromarray(ori.astype('uint8'), 'RGB')  
        img.save(predir+r'-ori.jpg')


    elif(dim==3):
        if(trans_case=='A'):
            assert(False)
        elif(trans_case=='B'):
            ori=scipy.fft.ifftn(freq)
        if(not prm_flt):
            assert(np.allclose(ori,data))



    
    return ori



def myfft3_legacy(data):
    print('[start myfft3_legacy]')
    freq=fp.rfft(fp.rfft(fp.rfft(data, axis=0),axis=1),axis=2)
    ori=fp.irfft(fp.irfft(fp.irfft(freq,axis=2),axis=1),axis=0)
    print('[end myfft3_legacy]')
    assert(np.allclose(ori,data))
    return ori




# Read in data file and transform
# predir=r'C:\Users\123\Pictures\\'
# picname=r'97049-3840x2160-rain-window-wallpaper-image-desktop-4k.jpg'
# picname=r'screenshot-1.png'
# picname=r'lowsig.png'
# picname=r'highsig.png'
# data = np.array(Image.open(predir+picname).convert('L'))
# prm_flt=0
# threslow_ratio=0.0
# threshigh_ratio=1.0
# useratio=1
# dim=2
# trans_case='B'
# flt_case='B'
# myfft(data=data,idx=0)
# exit(0)
# freq = im2freq(data)

# remmax = lambda x: x/x.max()
# remmin = lambda x: x - np.amin(x, axis=(0,1), keepdims=True)
# touint8 = lambda x: (remmax(remmin(x))*(256-1e-4)).astype(int)
# temp=touint8(freq)

# temp=np.repeat(freq[np.newaxis, ...], 3, axis=0)   #know
# temp=np.swapaxes(temp,0,1)
# temp=np.swapaxes(temp,1,2)

# img=Image.fromarray(temp.astype('uint8'), 'RGB')  
# img.save(predir+r'-freq.jpg')












# arr2im(touint8(freq), predir+r'freq.png')





# predir=r"D:\\CODE\\CCONV RES\\csm_mp300_50kexample_long2z+2\\"
# for i in range(0,200):
#     np.save(predir+"low-sfcurl-"+str(i)+".npy",\
#             myfft(np.load(predir+"sfcurl-"+str(i)+".npy"),i))
    
#     np.save(predir+"low-sfvel-"+str(i)+".npy",\
#             myfft(np.load(predir+"sfvel -"+str(i)+".npy"),i))


# -----------------------------------------------------------------------------------


predir=r"D:\\CODE\\CCONV RES\\csm_mp300_50kexample_long2z+2\\"
lv=250
rv=250
slicey=2
threslow=110
threshigh=140
threslow=90
threshigh=120



# predir=r'D:\CODE\CCONV RES\mix\y0_csm_mp300emit_large\y0_csm_mp300emit_large\\'
# lv=566
# rv=566
# slicey=5
# threslow=200
# threshigh=300
# prm_flt=1
# prm_vecnum=3


predir=r'D:/CODE/CCONV RES/mix/_csm300_1111_50kmc_ball_2velx_0602/_csm300_1111_50kmc_ball_2velx_0602/'
predir=r'D:/CODE/CCONV RES/mix/_csm_df300_1111_50kmc_ball_2velx_0602/_csm_df300_1111_50kmc_ball_2velx_0602/'
predir=r'D:/CODE/MCVSPH-FORK/specific_mt_1_output/'
predir=r'D:/CODE/MCVSPH-FORK/specific_df_1_output/'
# predir=r'D:/CODE/MCVSPH-FORK/specific_mp_1_output/'
# predir=r'D:/CODE/CCONV RES/mix/emin_b_mc_ball_2velx_0602/emin_b_mc_ball_2velx_0602/'
lv=1
rv=200
slicey=4
threslow=0
threshigh=100
prm_flt=0
prm_vecnum=1

# -----------------------------------------------------------------------------------
for i in tqdm(range(lv,rv+1)):
    if(prm_vecnum==1):
        data=np.load(predir+"vel-mat-"+str(i)+".npy")#L W H
        oris=myfft(data,i)
        
    elif(prm_vecnum==3):
        data=np.load(predir+"vel3-mat-"+str(i)+".npy")#L W H vecnum

        print('[data vel3 mean]'+str(np.mean(data[:,:,:,0])))

        orix=myfft(data[:,:,:,0],i).flatten()
        oriy=myfft(data[:,:,:,1],i).flatten()
        oriz=myfft(data[:,:,:,2],i).flatten()

        
        oris=np.stack((orix,oriy,oriz)).T

        print('[ori vel3 mean]'+str(np.mean(np.real(orix))))
        print('[ori vel3 mean]'+str(np.max(np.real(orix))))



    else:
        assert(False)
    visualRes(oris,idx=i)



 


# k=np.random.rand(3,5,7)
# print(k.shape)
# assert(np.allclose(fp.ifftn(fp.fftn(k)),k))
# print(fp.fftn(k).shape)

