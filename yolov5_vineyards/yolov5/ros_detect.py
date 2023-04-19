import argparse
import os
import sys
from pathlib import Path

import cv2

#Implementació amb ros
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

#model
from models.common import DetectMultiBackend
import torch
import torch.backends.cudnn as cudnn
from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync
import numpy as np


FILE = Path(__file__).resolve()  		#get the parent directory of the directory containing the script 
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

def identification(img):
	weights= ROOT / 'best.pt'
	data=ROOT / 'data/coco128.yaml'  # dataset.yaml path
	half=False  # use FP16 half-precision inference
	dnn=False   # use OpenCV DNN for ONNX inference
	device=''    # cuda device, i.e. 0 or 0,1,2,3 or cpu
	device = select_device(device)
	imgsz=(840, 640)  # inference size (height, width)
	augment=False,  # augmented inference
	visualize=False,  # visualize features
	#cudnn.benchmark = True  # set True to speed up constant image size inference
	
	#print('aqui va el model') 
	#load model
	model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
	stride, names, pt = model.stride, model.names, model.pt
	imgsz = check_img_size(imgsz, s=stride)  # check image size
	#print('model ', model)
	#dataset = LoadImages(1, img_size=1, stride=stride, auto=pt)
	#inference:
	model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
	dt, seen = [0.0, 0.0, 0.0], 0
	#això és del codi incial, prové de la classe LoadStreams, 
	#im=114*np.ones((1,3,320,416), dtype= np.int8)
	#im = torch.from_numpy(im).to(device)
	#im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
	#im /= 255  # 0 - 255 to 0.0 - 1.0
	#print('im ', im)
	#b, ch, h, w = img.shape
	#img=(1)+img
	print('sghape im ', img.shape)
	
	pred = model(img, augment=augment, visualize=visualize)
    
def callback_img(data):
	print('imgr received')
	bridge = CvBridge()
	img = bridge.imgmsg_to_cv2(data, "bgr8") #results in numpy
	identification(img)
	

def main():
    rospy.init_node('image_converter', anonymous=False)
    sub_img = rospy.Subscriber("/camera/color/image_raw",Image,callback_img)
    
    while True:
    	rospy.spin()


if __name__ == "__main__":
    
    main()
