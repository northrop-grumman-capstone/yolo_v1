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
from scipy.misc import imread, imresize
import matplotlib.pyplot as plt

class VotTrainDataset(data.Dataset):
    def __init__(self, videoDir, annotDir, img_size, S, B, C, transforms):
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


        bbox = []
        label = []
        for file in os.listdir(annotDir):
            filename = annotDir+file
            infile = open(filename,'rb')
            videoAnnot = pickle.load(infile)
            infile.close()
            for i,value in enumerate(videoAnnot):
                self.file_names.append(file[:-7]+"/"+str(i)+".npy")
                if len(value) != 0:
                    label.append(int(value[0][0]))
                    bbox.append([float(i) for i in value[0][1]])
                else:
                    label.append(-1)
                    bbox.append([float(-1) for i in range(4)])
                self.bboxes.append(torch.Tensor(bbox))
                self.labels.append(torch.IntTensor(label))
        self.n_data = len(self.labels)


    def __getitem__(self, index):
        bbox = self.bboxes[index].clone()
        label = self.labels[index].clone()

        img = Image.fromarray(np.load(os.path.join(self.videoDir, self.file_names[index])))

        width, height = img.size

        img = img.resize((self.img_size, self.img_size)) 
        bbox = bbox / torch.Tensor([width, height, width, height])# * self.img_size
        target = self.encode_target(bbox, label)
        transform = transforms.Compose(self.transforms)
        img = transform(img)
        return img, target

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
        x_index = (x_center / (1 / float(self.S))).ceil()-1
        y_index = (y_center / (1 / float(self.S))).ceil()-1
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


# def main():
#     img_size = 224
#     file = 'vot2017_train.txt'
#     img_folder = './vot2015'
#     train_dataset = VotTrainDataset(img_folder=img_folder, file=file, img_size=224, S=7, B=2, C=20, transforms=[transforms.ToTensor()])
#     #img, target = train_dataset.__getitem__(0)

#     train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)
#     train_iter = iter(train_loader)
#     img, target = next(train_iter)
#     for i in range(7):
#         for j in range(7):
#             if target[0,i,j,4] != 0:
#                 print(i,j)
#                 print(target[0,i,j])

#     # print(img.size())
#     # print(type(img))
#     # print(type(target))
#     img, target = next(train_iter)

#     print(img.size())
#     print(target.size())
#     img, target = next(train_iter)
#     print(img.size())
#     print(target.size())



# if __name__ == '__main__':
#     main()

