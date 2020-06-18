
import torch
import torchvision
import argparse
import cv2
import numpy as np
import sys
sys.path.append('./')
import coco_name
import random

def get_args():
    parser = argparse.ArgumentParser(description='Pytorch Faster-rcnn Detection')
 
    parser.add_argument('--image_path', type=str, default='D:/COCO/000000000722.jpg', help='image path')
    parser.add_argument('--model', default='fasterrcnn_shufflenet', help='model')
    parser.add_argument('--dataset', default='coco', help='model')
    parser.add_argument('--score', type=float, default=0.5, help='objectness score threshold')
    parser.add_argument('--model_path', default='D:/COCO/detection/Shufflenet_modle/model_20.pth', help='resume from checkpoint')
    args = parser.parse_args()
 
    return args
 
def random_color():
    b = random.randint(0,255)
    g = random.randint(0,255)
    r = random.randint(0,255)
 
    return (b,g,r)
 
def main():
    args = get_args()
    input = []
    num_classes = 91
    names = coco_name.names
        
    # Model creating
    print("Creating model")    
    model = torchvision.models.detection.__dict__[args.model](num_classes=num_classes, pretrained=False)

    # model = fasterrcnn_resnet50_fpn(num_classes=num_classes, pretrained_backbone=False)
    print(("=> loading checkpoint '{}'".format(args.model_path)))
    checkpoint = torch.load(args.model_path) #, map_location='cpu'
    model.load_state_dict(checkpoint['model'])#['model']
    model = model.cuda()
 
    model.eval()
 
    src_img = cv2.imread(args.image_path)
    img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img/255.).permute(2,0,1).float().cuda()
    input.append(img_tensor)
    out = model(input)
    boxes = out[0]['boxes']
    labels = out[0]['labels']
    scores = out[0]['scores']
 
    for idx in range(boxes.shape[0]):
        if scores[idx] >= args.score:
            x1, y1, x2, y2 = boxes[idx][0], boxes[idx][1], boxes[idx][2], boxes[idx][3]
            name = names.get(str(labels[idx].item()))
            cv2.rectangle(src_img,(x1,y1),(x2,y2),random_color(),thickness=2)
            score = format(scores[idx] , '.2f')
            text = '{} , {}' .format(name,score)
            cv2.putText(src_img, text=text, org=(x1, y1+10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                fontScale=0.5, thickness=1, lineType=cv2.LINE_AA, color=(0, 0, 255))

    # cv2.namedWindow('result',0)
    cv2.imshow('result',src_img)
    cv2.waitKey()
    cv2.destroyAllWindows()
    return
 
    
 
if __name__ == "__main__":
    main()