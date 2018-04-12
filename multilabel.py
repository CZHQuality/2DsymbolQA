#coding=utf-8
#This code is used to generate the new quality grade database using the direct grade obtained from 2DTG, and with multi labels
#The meaning of each label represents:  labelA Final Quality Score(0-4) - labelB Distortion Type(0-5) - labelC Decoded or Undecoded(0 or 1) 
# - labelD axial_nonuniformity(0-4) - labelE grid_nonuniformity(0-4) - labelF unused_error_corr(0-4) - labelG fixed_patt_damage(0-4) - labelH modulation grade(0-4) - labelI print_growth(0-4) 


import os
import numpy as np
import cv2
import math
import warnings
import pdb
from matplotlib import pyplot as plt
warnings.filterwarnings("ignore")

index = 0
index_0 = 490 #XXXX how to sort? 不要让GrNu覆盖AxNu 并看看如何让index的顺序更直观
index_1 = 238 #XXXX
index_2 = 524
index_3 = 1629
index_4 = 119

root_path = '/home/chezhaohui/TestImages/SJTU_2D_Datamatrix_Dataset/'
sv_path = '/home/chezhaohui/TestImages/DATAMultiLabel/'

root_path_1 = root_path + 'AxNu/'
root_path_2 = root_path + 'GrNu/'
root_path_3 = root_path + 'MotionBlur/'
root_path_4 = root_path + 'SpeckleNoise/'
root_path_5 = root_path + 'Thick/'
root_path_6 = root_path + 'Thin/'

sv_path_0 = sv_path + '0/'
sv_path_1 = sv_path + '1/'
sv_path_2 = sv_path + '2/'
sv_path_3 = sv_path + '3/'
sv_path_4 = sv_path + '4/'

for filename in os.listdir(root_path_6): #XXXXX
    input_img_path = root_path_6 + filename #XXXX
    # print(input_img_path)
    command = './My_Score3.out' + ' ' + input_img_path
    pipe = os.popen(command)
    labelVector = pipe.read()
    # print(labelVector)
    pipe.close
    labelA = labelVector[14:15]
    labelB = str(5) #XXXX 0:AxNu, 1:GrNu, 2:MB, 3:SN, 4:Thick, 5:Thin XXXX
    labelC = labelVector[16:17]
    labelD = labelVector[18:19]
    labelE = labelVector[20:21]
    labelF = labelVector[22:23]
    labelG = labelVector[24:25]
    labelH = labelVector[26:27]
    labelI = labelVector[28:29]

    temp_img = cv2.imread(input_img_path)

    flag = int(labelA) #Final Quality Score

    if(flag == 0):
        index_0 += 1 
        cv2.imwrite(sv_path_0 + labelA + '-' + labelB + '-' + labelC + '-'
                    + labelD + '-' + labelE + '-' + labelF + '-' + labelG
                    + '-' + labelH + '-' + labelI + '-' + str(index_0)
                    + '.jpg', temp_img)
    if(flag == 1):
        index_1 += 1
        cv2.imwrite(sv_path_1 + labelA + '-' + labelB + '-' + labelC + '-'
                    + labelD + '-' + labelE + '-' + labelF + '-' + labelG
                    + '-' + labelH + '-' + labelI + '-' + str(index_1)
                    + '.jpg', temp_img)
    if(flag == 2):
        index_2 += 1
        cv2.imwrite(sv_path_2 + labelA + '-' + labelB + '-' + labelC + '-'
                    + labelD + '-' + labelE + '-' + labelF + '-' + labelG
                    + '-' + labelH + '-' + labelI + '-' + str(index_2)
                    + '.jpg', temp_img)
    if(flag == 3):
        index_3 += 1
        cv2.imwrite(sv_path_3 + labelA + '-' + labelB + '-' + labelC + '-'
                    + labelD + '-' + labelE + '-' + labelF + '-' + labelG
                    + '-' + labelH + '-' + labelI + '-' + str(index_3)
                    + '.jpg', temp_img)
    if(flag == 4):
        index_4 += 1
        cv2.imwrite(sv_path_4 + labelA + '-' + labelB + '-' + labelC + '-'
                    + labelD + '-' + labelE + '-' + labelF + '-' + labelG
                    + '-' + labelH + '-' + labelI + '-' + str(index_4)
                    + '.jpg', temp_img)

print(index_0, index_1, index_2, index_3, index_4)
    


