import warnings
import os
import shutil
# 忽视警告
warnings.filterwarnings('ignore')

import cv2
from PIL import Image

import torch

from torch_py.FaceRec import Recognition


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 确认我们的电脑支持CUDA，然后显示CUDA信息：
#print(device)


pnet_path = "./torch_py/MTCNN/weights/pnet.npy"
rnet_path = "./torch_py/MTCNN/weights/rnet.npy"
onet_path = "./torch_py/MTCNN/weights/onet.npy"

image_folder = 'E:/dataset/oulu/trainset'
save_folder = 'E:/dataset/oulu/train_face'
if not os.path.exists(save_folder):
    os.mkdir(save_folder)

resolution_inp = 256
resolution_op = 256

first_dir = os.listdir(image_folder)
num = 0
for img in first_dir:
    # 二级目录绝对路径
    if img.split(".")[-1] == 'jpg' or img.split(".")[-1] == 'JPG':
        path_image = image_folder + '/' + str(img)
        #if int((img.split(".")[0]).split("_")[-2]) == 1:

        image = Image.open(path_image)
        #image = cv2.imread(path_image)
        num = num + 1
        print(num, str(img))

        recognize = Recognition()
        #draw = recognize.face_recognize(img)
        #plot_image(draw)
        if not recognize.crop_faces(image):
            #face_img = image
            print("Failed!!")
            continue
            #plot_image(image)
        else:
            face_img = recognize.crop_faces(image)[0]
            #plot_image(face_img)

        save_path = save_folder + '/' + str(img)
        face_img.save(save_path)

        #cv2.imwrite(save_path, face_img)

