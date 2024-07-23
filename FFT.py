from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import scipy.fft
import scipy.fftpack as fp      #legacy
import scipy

np.random.seed(1234)

# https://stackoverflow.com/questions/38476359/fft-on-image-with-python     answer1

predir=r'C:\Users\123\Pictures\\'


## Functions to go from image to frequency-image and back
im2freq = lambda data: fp.rfft(fp.rfft(data, axis=0),
                               axis=1)
#迭代FFT，先对一个轴做，然后再对另一个轴做



freq2im = lambda f: fp.irfft(fp.irfft(f, axis=1),
                             axis=0)




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

    plt.savefig(predir+'amp.png')
    plt.close()
    

def AmplitudeB(freq,dim=2):
    print('drawing amplituide...')
    plotdict={}
    centerx=freq.shape[0]/2
    centery=freq.shape[1]/2

    if(dim==3):
        centerz=freq.shape[2]/2



    global predir
    for index, value in np.ndenumerate(freq):
        if(dim==2):
            dis=abs(index[0]-centerx)+abs(index[1]-centery)
        elif(dim==3):
            dis=abs(index[0]-centerx)+abs(index[1]-centery)+abs(index[2]-centerz)

        dis=int(dis)
        if dis in plotdict:
            plotdict[dis]+=value
        else:
            plotdict[dis]=value
        # print('index\t'+str(index))

    # plt.xlim((0,10000))
    
    plt.plot(list(plotdict.keys()),list(plotdict.values()))
    if(dim==2):
        plt.savefig(predir+'amp.png')
    elif(dim==3):
        plt.savefig(predir+'amp3d.png')
    plt.close()
    print('drawing done')

    

flt=None
#caseB的频谱图是以原点为中心对称的，所以flt也是先保留中间区域的
def makefltB(freq,threslow=0.9,threshigh=1.1):
    centerx=freq.shape[0]/2
    centery=freq.shape[1]/2
    print('[making filter2]')
    global flt
    flt=np.zeros_like(freq)

    for index, value in np.ndenumerate(flt):#know
        # print(index)
        # print(index[0])
        dis=abs(index[0]-centerx)+abs(index[1]-centery)
        if(    dis/freq.shape[0]<threshigh \
           and dis/freq.shape[0]>threslow ):
            flt[index]=1
            
    print(np.sum(flt))
    print(flt.shape[0]*flt.shape[1])

def makefltA(freq,threslow=0,threshigh=1.0):
    #以扇形区域删减频谱图。适用于caseA.max thers:1.42
    thres=0.002
    thres=0
    print('[making filter]')
    global flt
    flt=np.zeros_like(freq)
    
    for index, value in np.ndenumerate(flt):#know
        # print(index)
        # print(index[0])
        r2=index[0]**2+index[1]**2
        if(    r2 <=   (threshigh*freq.shape[0])**2\
           and r2 >=   (threslow *freq.shape[0])**2 ):
            flt[index]=1

    print(np.sum(flt))
    print(flt.shape[0]*flt.shape[1])
hasflt=0


## Helper functions to rescale a frequency-image to [0, 255] and save




def myfft(data,idx,dim=2):
    
    if(dim==2):
        #2D method
        # freq=fp.rfft(fp.rfft(data, axis=0),axis=1)        #caseA
        # freq=fp.fft2(data)
        # freq=scipy.fft.rfft2(data)        #输出的尺寸不对
        freq=scipy.fft.fft2(data)#shape gridnum gridnum     #caseB

    elif(dim==3):
        #3D method
        # print(np.sum(np.isnan(data)))
        # print(np.max(data))
        # print(np.min(data))
        # print(np.mean(data))
        # print(data.dtype)
        # print(data[0,0,0])

        data=data.astype('float64')
        freq=scipy.fft.fftn(data)

      

     
        

 
    if(dim==2):
        # Read in data file and transform
        remmax = lambda x: x/x.max()
        remmin = lambda x: x - np.amin(x, axis=(0,1), keepdims=True)
        touint8 = lambda x: (remmax(remmin(x))*(256-1e-4)).astype(int)
        temp=touint8(freq)
        temp=np.repeat(freq[np.newaxis, ...], 3, axis=0)   #know
        temp=np.swapaxes(temp,0,1)
        temp=np.swapaxes(temp,1,2)
        img=Image.fromarray(temp.astype('uint8'), 'RGB')  
        img.save(predir+r'-freq.jpg')


    if((idx==133 and dim==2)\
       or(idx==30 and dim==3)):#223,791,297,30
        AmplitudeB(freq,dim=dim)

    if(dim==2):
        global flt,hasflt
        if(not hasflt):
            makefltB(freq)
            hasflt=1
        freq*=flt


    if(dim==2):
        # ori=fp.irfft(fp.irfft(freq,axis=1),axis=0)
        # ori=fp.ifft2(freq)
        ori=scipy.fft.ifft2(freq)
    elif(dim==3):
        ori=scipy.fft.ifftn(freq)
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
# data = np.array(Image.open(predir+r'97049-3840x2160-rain-window-wallpaper-image-desktop-4k.jpg'))
# # freq = im2freq(data)
# freq = scipy.fft.fft2(data)
# remmax = lambda x: x/x.max()
# remmin = lambda x: x - np.amin(x, axis=(0,1), keepdims=True)
# touint8 = lambda x: (remmax(remmin(x))*(256-1e-4)).astype(int)
# temp=touint8(freq)

# temp=np.repeat(freq[np.newaxis, ...], 3, axis=0)   #know
# temp=np.swapaxes(temp,0,1)
# temp=np.swapaxes(temp,1,2)

# img=Image.fromarray(temp.astype('uint8'), 'RGB')  
# img.save(predir+r'-freq.jpg')
# if(not hasflt):
#     makefltA(freq)
#     hasflt=1
# freq*=flt
# # back = freq2im(freq)
# back = scipy.fft.ifft2(freq)

# # Make sure the forward and backward transforms work!
# # assert(np.allclose(data, back))#know
# print(back.shape)
# back=np.repeat(back[np.newaxis, ...], 3, axis=0)   #know
# back=np.swapaxes(back,0,1)
# back=np.swapaxes(back,1,2)
# print(back.shape)
# # out.putdata(map(tuple, back.reshape(-1, 3)))
# img=Image.fromarray(back.astype('uint8'), 'RGB')  
# img.save(predir+r'-ori.jpg')
# exit(0)


# print(type(freq))
# print(freq.shape)
# exit(0)







# arr2im(touint8(freq), predir+r'freq.png')





# predir=r"D:\\CODE\\CCONV RES\\csm_mp300_50kexample_long2z+2\\"
# for i in range(0,200):
#     np.save(predir+"low-sfcurl-"+str(i)+".npy",\
#             myfft(np.load(predir+"sfcurl-"+str(i)+".npy"),i))
    
#     np.save(predir+"low-sfvel-"+str(i)+".npy",\
#             myfft(np.load(predir+"sfvel -"+str(i)+".npy"),i))




predir=r"D:\\CODE\\CCONV RES\\csm_mp300_50kexample_long2z+2\\"
for i in range(1,900):
    myfft(np.load(predir+"velnorm-"+str(i)+".npy"),i,dim=3)
    
 


# k=np.random.rand(3,5,7)
# print(k.shape)
# assert(np.allclose(fp.ifftn(fp.fftn(k)),k))
# print(fp.fftn(k).shape)

