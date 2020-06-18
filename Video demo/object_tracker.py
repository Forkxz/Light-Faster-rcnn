from models import *
from utils import *

import os, sys, time, datetime, random
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable
import cv2
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
conf_thres=0.8
nms_thres=0.4
model_name = "maskrcnn_resnet50_fpn"
model_path = "D:/COCO/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth"
# D:/COCO/detection/Shufflenet_modle/model_23.pth
# D:/COCO/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth
# Load model and weights
# Model creating
print("Creating model")    
# model = torchvision.models.detection.__dict__['fasterrcnn_shufflenet'](num_classes=91, pretrained=False)
model = torchvision.models.detection.__dict__[model_name](num_classes=91, pretrained=False)


print(("=> loading checkpoint '{}'".format(model_path)))
checkpoint = torch.load(model_path) #, map_location='cuda'
model.load_state_dict(checkpoint)#['model']
model = model.cuda()
model.eval()
Tensor = torch.cuda.FloatTensor

def random_color():
    b = random.randint(0,255)
    g = random.randint(0,255)
    r = random.randint(0,255)
 
    return (b,g,r)

def detect_image(img):
    img_transforms = transforms.Compose([transforms.ToTensor()])
    # # convert image to Tensor
    image_tensor =img_transforms(img).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input_img = Variable(image_tensor.type(Tensor))
    # run inference on the model and get detections
    with torch.no_grad():
        detections = model(input_img)
    return detections[0]

def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask >= alpha,
                                  image[:, :, c] *
                                  0.5 + 0.5 * color[c],
                                  image[:, :, c])
    return image

videopath = 'D:/COCO/detection/000.mp4'
vid = cv2.VideoCapture(videopath)


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

    if detections is not None:
        boxes = detections['boxes']
        labels = detections['labels']
        scores = detections['scores']
        masks = detections['masks']
        for idx in range(boxes.shape[0]):
            color = random_color()
            if scores[idx] >= 0.8:
                x1, y1, x2, y2 = boxes[idx][0], boxes[idx][1], boxes[idx][2], boxes[idx][3]
                name = names.get(str(labels[idx].item()))
                cv2.rectangle(frame,(x1,y1),(x2,y2),color,thickness=2)
                score = format(scores[idx] , '.2f')
                text = '{} , {}' .format(name,score)
                # Mask
                mask = masks[idx,0,:, :].cpu()
                frame = apply_mask(frame, mask, color)
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
