import cv2
import os
import torch
import pickle
import numpy as np
from PIL import Image
import deepdish as dd
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

class FramesDataset(data.Dataset):
    def __init__(self, videoDir, annotDir, img_size, S, B, C, transforms, training=True):
        self.videoDir = videoDir
        self.annotDir = annotDir
        self.file_names = []
        self.img_size = img_size
        self.S = S
        self.B = B
        self.C = C
        self.transforms = transforms
        self.bboxes = []
        self.labels = []
        self.training = training

        for file in os.listdir(annotDir):
            filename = annotDir+file
            infile = open(filename,'rb')
            videoAnnot = pickle.load(infile)
            infile.close()
            for i,value in enumerate(videoAnnot):
                bbox = []
                label = []
                for j in range(len(value)):
                    if(j!=0): break # original code only used one annotation, remove later if it works with more
                    self.file_names.append(file[:-7]+"/"+str(i)+".jpeg")
                    label.append(int(value[j][0]))
                    if(self.training):
                        # pickle files have [xmin, xmax, ymin, ymax] between 0 and 1
                        # this expected [xcenter, ycenter, height, width] in img coords right here
                        # but I changed later code, so it expects it between 0 and 1
                        bbox.append([(value[j][1][0]+value[j][1][1])/2, (value[j][1][2]+value[j][1][3])/2, value[j][1][1]-value[j][1][0], value[j][1][3]-value[j][1][2]])
                    else:
                        bbox.append(value[j][1])
                if(len(value)!=0):
                    self.bboxes.append(torch.Tensor(bbox))
                    self.labels.append(torch.IntTensor(label))
        self.n_data = len(self.labels)


    def __getitem__(self, index):
        bbox = self.bboxes[index].clone()
        label = self.labels[index].clone()

        img = Image.open(os.path.join(self.videoDir, self.file_names[index]))

        width, height = img.size

        img = img.resize((self.img_size, self.img_size))
        # the following line resized bboxes to between 0 and 1, but ours are already like that
        #bbox = bbox / torch.Tensor([width, height, width, height])# * self.img_size
        if(self.training):
            target = self.encode_target(bbox, label)
        transform = transforms.Compose(self.transforms)
        img = transform(img)
        return img, target if self.training else (label, bbox)

    def encode_target(self, bbox, label):
        """

        :param bbox: [xc,yc,w,h] coordinates in the top left and bottom right separately
        :param label: class label
        :return: [normalized_xc,normalized_yc,sqrt(normalized_w),sqrt(normalized_h)]
        """
        n_elements = self.B * 5 + self.C
        n_bbox = len(label)
        target = torch.zeros((self.S, self.S, n_elements))
        class_info = torch.zeros((n_bbox, self.C))
        for i in range(n_bbox):
            class_info[i, label[i]] = 1
        w = bbox[:,2]
        w_sqrt = torch.sqrt(w)
        x_center = bbox[:,0]
        h = bbox[:,3]
        h_sqrt = torch.sqrt(h)
        y_center = bbox[:,1]

        x_index = torch.clamp((x_center / (1 / float(self.S))).ceil()-1, 0, self.S-1)
        y_index = torch.clamp((y_center / (1 / float(self.S))).ceil()-1, 0, self.S-1)
        # bounding box centers are offsets from grid, not absolute, may remove if performs poorly
        #x_center = torch.clamp((x_center / (1 / float(self.S))), 0, self.S-1) - x_index
        #y_center = torch.clamp((y_center / (1 / float(self.S))), 0, self.S-1) - y_index

        c = torch.ones_like(x_center)
        # set w_sqrt and h_sqrt directly

        box_block = torch.cat((x_center.view(-1,1), y_center.view(-1,1), w_sqrt.view(-1,1), h_sqrt.view(-1,1), c.view(-1,1)), dim=1)
        box_info = box_block.repeat(1, self.B)
        target_infoblock = torch.cat((box_info, class_info), dim=1)

        for i in range(n_bbox):
            target[int(x_index[i]),int(y_index[i])] = target_infoblock[i].clone()
        return target

    def __len__(self):
        return self.n_data
