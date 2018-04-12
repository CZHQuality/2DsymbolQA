from __future__ import print_function
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import numpy
import cv2
import random

list = os.listdir("/media/ubuntu/CZHhy/BarcodeQA/TIEcode/DATAMultiLabel/Aug/") #change as your own path
random.shuffle(list) 
print("list is:", list)
fh = open("/media/ubuntu/CZHhy/BarcodeQA/TIEcode/DATAMultiLabel/list.txt", 'w')
for line in list:
    fh.write(line+'\n')
fh.close()