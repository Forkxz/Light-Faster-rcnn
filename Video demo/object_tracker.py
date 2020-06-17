from models import *
from utils import *

import os, sys, time, datetime, random
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable

from PIL import Image
import torch
import torchvision
import argparse
import cv2
import numpy as np
import sys
sys.path.append('./')
import random

names = {'0': 'background', '1': 'person', '2': 'bicycle', '3': 'car', '4': 'motorcycle', '5': 'airplane', '6': 'bus', '7': 'train', '8': 'truck', '9': 'boat', '10': 'traffic light', '11': 'fire hydrant', '13': 'stop sign', '14': 'parking meter', '15': 'bench', '16': 'bird', '17': 'cat', '18': 'dog', '19': 'horse', '20': 'sheep', '21': 'cow', '22': 'elephant', '23': 'bear', '24': 'zebra', '25': 'giraffe', '27': 'backpack', '28': 'umbrella', '31': 'handbag', '32': 'tie', '33': 'suitcase', '34': 'frisbee', '35': 'skis', '36': 'snowboard', '37': 'sports ball', '38': 'kite', '39': 'baseball bat', '40': 'baseball glove', '41': 'skateboard', '42': 'surfboard', '43': 'tennis racket', '44': 'bottle', '46': 'wine glass', '47': 'cup', '48': 'fork', '49': 'knife', '50': 'spoon', '51': 'bowl', '52': 'banana', '53': 'apple', '54': 'sandwich', '55': 'orange', '56': 'broccoli', '57': 'carrot', '58': 'hot dog', '59': 'pizza', '60': 'donut', '61': 'cake', '62': 'chair', '63': 'couch', '64': 'potted plant', '65': 'bed', '67': 'dining table', '70': 'toilet', '72': 'tv', '73': 'laptop', '74': 'mouse', '75': 'remote', '76': 'keyboard', '77': 'cell phone', '78': 'microwave', '79': 'oven', '80': 'toaster', '81': 'sink', '82': 'refrigerator', '84': 'book', '85': 'clock', '86': 'vase', '87': 'scissors', '88': 'teddybear', '89': 'hair drier', '90': 'toothbrush'}

class_path='D:/COCO/pytorch_objectdetecttrack-master/config/coco.names'
img_size=416
conf_thres=0.8
nms_thres=0.4
# Load model and weights
# Model creating
print("Creating model")    
model = torchvision.models.detection.__dict__['fasterrcnn_shufflenet'](num_classes=91, pretrained=False)

# model = fasterrcnn_resnet50_fpn(num_classes=num_classes, pretrained_backbone=False)
print(("=> loading checkpoint '{}'".format('D:/COCO/detection/Shufflenet_modle/model_23.pth')))
checkpoint = torch.load('D:/COCO/detection/Shufflenet_modle/model_23.pth') #, map_location='cuda'
model.load_state_dict(checkpoint['model'])#['model']
model = model.cuda()
 
model.eval()
# classes = utils.load_classes(class_path)
Tensor = torch.cuda.FloatTensor
def random_color():
    b = random.randint(0,255)
    g = random.randint(0,255)
    r = random.randint(0,255)
 
    return (b,g,r)
def detect_image(img):
    # scale and pad image
    # imw = img.size[0] 
    # imh = img.size[1]
    img_transforms = transforms.Compose([transforms.ToTensor()])
    # # convert image to Tensor
    image_tensor =img_transforms(img).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input_img = Variable(image_tensor.type(Tensor))
    # run inference on the model and get detections
    with torch.no_grad():
        detections = model(input_img)
        # detections = utils.non_max_suppression(detections, 80, conf_thres, nms_thres)
    return detections[0]

videopath = 'D:/COCO/detection/000.mp4'

import cv2
from sort import *
colors=[(255,0,0),(0,255,0),(0,0,255),(255,0,255),(128,0,0),(0,128,0),(0,0,128),(128,0,128),(128,128,0),(0,128,128)]

vid = cv2.VideoCapture(videopath)
mot_tracker = Sort() 

cv2.namedWindow('Stream',cv2.WINDOW_NORMAL)
cv2.resizeWindow('Stream', (800,600))

fourcc = cv2.VideoWriter_fourcc(*'XVID')
ret,frame=vid.read()
vw = frame.shape[1]
vh = frame.shape[0]
print ("Video size", vw,vh)
outvideo = cv2.VideoWriter(videopath.replace(".mp4", "-det.mp4"),fourcc,20.0,(vw,vh))

frames = 0
starttime = time.time()
while(True):
    ret, frame = vid.read()
    if not ret:
        break
    frames += 1
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pilimg = Image.fromarray(frame)
    detections = detect_image(pilimg)

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    img = np.array(pilimg)
    pad_x = max(img.shape[0] - img.shape[1], 0) * (img_size / max(img.shape))
    pad_y = max(img.shape[1] - img.shape[0], 0) * (img_size / max(img.shape))
    unpad_h = img_size - pad_y
    unpad_w = img_size - pad_x
    if detections is not None:
        boxes = detections['boxes']
        labels = detections['labels']
        scores = detections['scores']
        # objects = detections[boxes, scores]
        # tracked_objects = mot_tracker.update(objects)
        for idx in range(boxes.shape[0]):
            if scores[idx] >= 0.8:
                x1, y1, x2, y2 = boxes[idx][0], boxes[idx][1], boxes[idx][2], boxes[idx][3]
                name = names.get(str(labels[idx].item()))
                cv2.rectangle(frame,(x1,y1),(x2,y2),random_color(),thickness=2)
                score = format(scores[idx] , '.2f')
                text = '{} , {}' .format(name,score)
                cv2.putText(frame, text=text, org=(x1, y1+10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                    fontScale=0.5, thickness=1, lineType=cv2.LINE_AA, color=(0, 0, 255))
    cv2.imshow('Stream', frame)
    outvideo.write(frame)
    ch = 0xFF & cv2.waitKey(1)
    if ch == 27:
        break

totaltime = time.time()-starttime
print(frames, "frames", frames/totaltime, "FPS")
cv2.destroyAllWindows()
outvideo.release()
