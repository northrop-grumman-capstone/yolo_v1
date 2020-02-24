#!/usr/bin/env python3

import os
import numpy as np
import torch
import torchvision.transforms
import pickle
from PIL import Image
from mini import *
import utils

n_batch = 64

saved_network = "results/model_mini.pth"
model = YOLO_V1()
model.load_state_dict(torch.load(saved_network))
print("Loaded trained model")
model.eval()

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
#if torch.cuda.device_count() > 1:
#    print("Let's use", torch.cuda.device_count(), "GPUs!")
#    model = nn.DataParallel(model).cuda()
model.to(device)

anno_folder = "/media/trocket/27276136-d5a4-4943-825f-7416775dc262/home/trocket/data/train/annots/"
vid_folder = "/media/trocket/27276136-d5a4-4943-825f-7416775dc262/home/trocket/data/train/videos/"

toTensor = torchvision.transforms.ToTensor()

while(True):
	try:
		v = input("Enter video number: ")
		anno = utils.load_anno(os.path.join(anno_folder, v+".pickle"))
		frames = [None]*len(anno)
		bboxes = []
		for i in range(len(anno)):
			frames[i] = toTensor(Image.open(os.path.join(vid_folder, "{}/{}.jpeg".format(v, i))).resize((224,224))).to(device)
		with(torch.no_grad()):
			for i in range(0, len(anno), n_batch):
				imgs = torch.stack(frames[i:i+n_batch])
				yolout = model(imgs)
				bboxes += utils.yolo_to_bbox(yolout)
		utils.play_formatted(os.path.join(vid_folder, v), 200, annotations=bboxes, class_labels=True)
	except KeyboardInterrupt:
		break
	#except: pass
