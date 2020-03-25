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

class VideoDataset(data.Dataset):
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
            index = 0

            if(self.training): videoAnnot = self.interp(videoAnnot)
            else:
                for i in range(len(videoAnnot)):
                    if(len(videoAnnot[i])==0):
                        videoAnnot[i] = [(-1, [-1, -1, -1, -1])]

            while index < len(videoAnnot):
                bbox = []
                label = []
                fileNames = []
                while (len(bbox) < 8 or not self.training) and index < len(videoAnnot):
                    value = videoAnnot[index]
                    for j in range(len(value)):
                        if(j!=0): break # original code only used one annotation, remove later if it works with more
                        fileNames.append(file[:-7]+"/"+str(index)+".jpeg")
                        label.append(torch.IntTensor([int(value[j][0])]))
                        if(self.training):
                            # pickle files have [xmin, xmax, ymin, ymax] between 0 and 1
                            # this expected [xcenter, ycenter, height, width] in img coords right here
                            # but I changed later code, so it expects it between 0 and 1
                            bbox.append(torch.Tensor([[(value[j][1][0]+value[j][1][1])/2, (value[j][1][2]+value[j][1][3])/2, value[j][1][1]-value[j][1][0], value[j][1][3]-value[j][1][2]]]))
                        else:
                            bbox.append(torch.Tensor([value[j][1]]))
                    index += 1

                if((self.training and len(bbox) == 8) or not self.training):
                    self.file_names.append(fileNames)
                    self.bboxes.append(torch.stack(bbox))
                    self.labels.append(torch.stack(label))



    def __getitem__(self, index):
        bbox = self.bboxes[index].clone()
        label = self.labels[index].clone()

        images = []
        transform = transforms.Compose(self.transforms)

        for image in self.file_names[index]:
            imagePath = os.path.join(self.videoDir, image)
            img = Image.open(imagePath)
            width, height = img.size
            img = img.resize((self.img_size, self.img_size))
            img = transform(img)
            images.append(img)
        # the following line resized bboxes to between 0 and 1, but ours are already like that
        #bbox = bbox / torch.Tensor([width, height, width, height])# * self.img_size
        target = []
        for i in range(len(bbox)):
            if(self.training): target.append(self.encode_target(bbox[i], label[i]))
            else: target.append((label[i], bbox[i]))

        return torch.stack(images), torch.stack(target) if self.training else target

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

    def interp(self, annos):
        l1 = -1
        l2 = -1
        for i in range(len(annos)):
            if(len(annos[i])!=0):
                if(l1==-1):
                    l1 = i
                else:
                    if(l2!=-1): l1 = l2
                    l2 = i
            if(l2!=i or annos[l1][0][0]!=annos[l2][0][0] or l2-l1>5): continue
            for j in range(l1+1, l2):
                a = (annos[l1][0][0], [0,0,0,0])
                f = (j-l1)/(l2-l1)
                a[1][0] = f*annos[l2][0][1][0]+(1-f)*annos[l1][0][1][0]
                a[1][1] = f*annos[l2][0][1][1]+(1-f)*annos[l1][0][1][1]
                a[1][2] = f*annos[l2][0][1][2]+(1-f)*annos[l1][0][1][2]
                a[1][3] = f*annos[l2][0][1][3]+(1-f)*annos[l1][0][1][3]
                annos[j].append(a)
        return annos

    def __len__(self):
        return len(self.file_names)
