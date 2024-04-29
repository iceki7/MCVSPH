# import os
# import cv2
# from tqdm import tqdm     # python 进度条库


image_folder_dir = r"D:\CODE\MCVSPH-FORK\ball-2_output_img\\"
# fps = 24     # fps: frame per seconde 每秒帧数，数值可根据需要进行调整
# size = (640, 360)     # (width, height) 数值可根据需要进行调整
# fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')     # 编码为 mp4v 格式，注意此处字母为小写，大写会报错
# video = cv2.VideoWriter(image_folder_dir+"your_video_name.mp4", fourcc, fps, size, isColor=True)

# image_list = sorted([name for name in os.listdir(image_folder_dir) if name.endswith('.png')])     # 获取文件夹下所有格式为 jpg 图像的图像名，并按时间戳进行排序
# for image_name in tqdm(image_list):     # 遍历 image_list 中所有图像并添加进度条
# 	image_full_path = os.path.join(image_folder_dir, image_name)     # 获取图像的全路经
# 	image = cv2.imread(image_full_path)     # 读取图像
# 	video.write(image)     # 将图像写入视频

# video.release()
# cv2.destroyAllWindows()


# import cv2
# img = cv2.imread(image_folder_dir+r'000000.png')
# imgInfo = img.shape
# # 宽度和高度信息
# size = (imgInfo[1],imgInfo[0])
# print(size)

# # windows下使用DIVX
# fourcc = cv2.VideoWriter_fourcc(*'DIVX')
# # VideoWriter 参数1: 写入对象 参数
# videoWrite = cv2.VideoWriter('pic2video.avi',fourcc,5,size,True)
# # 写入对象 
# # 1 file name 2 可用编码器（苹果笔记本直接写-1） 3 帧率 4 size
# for i in range(0,100,4):
#     fileName = '{0:06d}'.format(i)+'.png'
#     img = cv2.imread(fileName)
#     videoWrite.write(img) # 写入方法 1 jpg data
# print('end!')



# import cv2
# import numpy as np

# fc = cv2.VideoWriter_fourcc(*"mp4v")
# video = cv2.VideoWriter("1.mp4", fc, 20, (981, 958))

# for idx in range(264,999,4):
#     # color = np.random.randint(0, 255, size=3)
#     img = cv2.imread(image_folder_dir+r'{0:06d}.png'.format(idx))
#     print(img.shape)
#     # print(type(img))

#     # if idx in [0, 2, 3]:  # only 3 frames will be in the final video
#     #     image = np.full((500, 500, 3), fill_value=color, dtype=np.uint8)
#     # else:
#     #     # slighly different size
#     #     image = np.full((400, 500, 3), fill_value=color, dtype=np.uint8)

#     video.write(img)




import cv2
import numpy as np
import glob

img_array = []
for filename in glob.glob(image_folder_dir+'*.png'):
    img = cv2.imread(filename)
    height, width, layers = img.shape
    if(height>981):
        continue
    size = (width,height)
    img_array.append(img)

fps=15
out = cv2.VideoWriter('project.avi',cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()