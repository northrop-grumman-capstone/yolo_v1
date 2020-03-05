#!/usr/bin/env python3

import os
import numpy as np
import torch
import torchvision.transforms
import pickle
import argparse
from collections import OrderedDict
from PIL import Image
import utils
import load_frames
import load_videos

model = None
device = None
recurrent = False #TODO actually use this variable

anno_format = "/media/trocket/27276136-d5a4-4943-825f-7416775dc262/home/trocket/data/{}/annots/"
vid_format = "/media/trocket/27276136-d5a4-4943-825f-7416775dc262/home/trocket/data/{}/videos/"

toTensor = torchvision.transforms.ToTensor()

def load_network(model_name, gpu): # gpu = 2 to use both, 0 and 1 for cuda:0 and cuda:1, -1 for cpu
	global model, device
	if("rnn" in model_name): # add additional when lstm and stuff are implemented
		import yolo_rnn_net
		model = yolo_rnn_net.YOLO_V1()
		recurrent = True
	elif("mini" in model_name):
		import mini
		model = mini.YOLO_V1()
	else:
		import network
		model = network.YOLO_V1()
	state_dict = OrderedDict()
	for k, v in torch.load(saved_network).items(): #in case model was saved with nn.DataParallel
		if(k.startswith("module.")): state_dict[k[7:]] = v
		else: state_dict[k] = v
	model.load_state_dict(state_dict)
	print("Loaded trained model")
	model.eval()
	device = torch.device("cuda:"+str(gpu) if torch.cuda.is_available() and gpu!=-1 else "cpu")
	if(torch.cuda.device_count() > 1 and gpu == 2):
	    print("Let's use", torch.cuda.device_count(), "GPUs!")
	    model = nn.DataParallel(model)
	model.to(device)

def run_interactive(vid_folder, anno_folder): #TODO add rnn support
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

def train(vid_folder, anno_folder, **kwargs): #TODO
	pass

def validate(vid_folder, anno_folder, **kwargs):
	n_batch = kwargs["batch_size"]
	workers = kwargs["workers"]
	iou_thresh = kwargs["iou_thresh"]
	conf_thresh = kwargs["conf_thresh"]
	multiclass = kwargs["multiclass"]
	classdiff = kwargs["classdiff"]
	results_file = kwargs["results_file"]
	if(results_file==""):
		results_file = os.path.join("results/", "result_"+os.splitext(os.basename(kwargs["model"]))[1]+".pickle")
	if(recurrent):
		dataset = VideoDataset(vid_folder, anno_folder, 224, 7, 2, 24, [toTensor], encode=False, split_video=False)
		if(n_batch==-1): n_batch = 1
	else:
		dataset = FramesDataset(vid_folder, anno_folder, 224, 7, 2, 24, [toTensor], encode=False)
		if(n_batch==-1): n_batch = 64
	loader = DataLoader(dataset, batch_size=n_batch, num_workers=workers)
	preds = [[] for x in range(24)] # [[(confidence, iou)]]
	total_true = [0 for x in range(24)]
	for i, (data, target) in enumerate(loader):
		data = data.to(device)
		if(recurrent): pass #TODO
		else:
			bboxes = utils.yolo_to_bbox(model(data), conf_thresh=conf_thresh, iou_thresh=iou_thresh, multiclass=multiclass, classdiff=classdiff)
			used = [False]*len(target)
			for box in target: total_true[box[0]] += 1
			for bbox in bboxes:
				m = (-1,0)
				for i in range(len(target)):
					if(used[i] or bbox[0]!=target[i][0]): continue
					iou = utils.iou(bbox, target[i][1])
					if(iou>m[1]): m = (i, iou)
				if(m[0]!=-1):
					used[m[0]] = True
					preds[bbox[0]].append((bbox[2], m[1]))
				else: preds[bbox[0]].append((bbox[2], 0))
			if(i%20==0): print("\rCompleted {}/{} batches".format(i, len(loader)))
	print("\nCompleted all inputs, calculating metrics...\n")
	results = {}
	ious = np.linspace(.5, .95, 10)
	for i in range(24):
		if(i==22): continue # Skip class None
		preds[i].sort(reverse=True)
		arr = np.array(preds[i])
		class_results = {}
		for iou in ious:
			pos_so_far = np.cumsum(arr>iou)
			precision = pos_so_far/np.array(range(1, arr.size+1))
			precision = np.maximum.accumulate(precision[::-1])[::-1]
			recall = pos_so_far/total_true[i]
			AP = np.sum(precision[1:]*np.diff(recall))+precision[0]*recall[0]
			iou_results = {}
			iou_results["AP"] = AP
			iou_results["Precision"] = precision[::int(precision.size)/200] # store interpolation for plotting
			iou_results["Recall"] = recall[::int(recall.size)/200] # store interpolation for plotting
			class_results["IOU "+str(iou)] = iou_results
		class_results["mAP"] = sum([item[1]["AP"] for item in class_results.items()])/10
		results[utils.classes[i]] = class_results
	results["mAP"] = sum([item[1]["mAP"] for item in results.items()])/23
	with open(results_file, 'wb+') as f:
		pickle.dump(results, f)


def main():
	parser = argparse.ArgumentParser(description="Run a model.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	subparsers = parser.add_subparsers(dest="command")
	parser.add_argument("-m", "--model", required=True, help="model to run")
	parser.add_argument("-g", "--gpu", default=1, type=int, help="which gpu to use")

	# For anything required by both training and metrics
	tm_parent_parser = argparse.ArgumentParser(add_help=False)
	tm_parent_parser.add_argument("-n", "--batch_size", type=int, default=-1, help="batch size, -1 for auto")
	tm_parent_parser.add_argument("-w", "--workers", default=4, type=int, help="number of worker threads")

	# For anything required by both interactive and metrics
	im_parent_parser = argparse.ArgumentParser(add_help=False)
	im_parent_parser.add_argument("-s", "--set", default="valid", help="dataset to use, train, valid, or test")
	im_parent_parser.add_argument("-i", "--iou_thresh", type=float, default=0.15, help="IOU threshold at which to suppress boxes")
	im_parent_parser.add_argument("-c", "--conf_thresh", type=float, default=0.25, help="confidence threshold for accepting boxes")
	im_parent_parser.add_argument("--multiclass", action="store_true", help="if all class labels meeting conf_thresh should be used")
	im_parent_parser.add_argument("--classdiff", action="store_false", help="if classes should be accounted for during non-max suppression")

	# For any arguments required for training
	train_parser = subparsers.add_parser("train", parents=[tm_parent_parser], help="train a model")

	# For any arguments required by interactive
	interactive_parser = subparsers.add_parser("interactive", parents=[im_parent_parser], help="run a model on training videos interactively")

	# For any arguments required by metrics
	metrics_parser = subparsers.add_parser("metrics", parents=[im_parent_parser, tm_parent_parser], help="compute mAP and other metrics on a validation/test set")
	metrics_parser.add_argument("-f", "--result_file", help="file to write out result pickle to")

	args = parser.parse_args()
	if("valid_set" in args and args["valid_set"] not in ["train", "test", "valid"]):
		print("Invalid choice of validation set, please choose train, valid, or test.")
		return

	load_network(args["model"], args["gpu"])
	if(args["command"]=="interactive"):
		run_interactive(vid_format.format(args["valid_set"]), anno_format.format(args["valid_set"]), **vars(args))
	elif(args["command"]=="train"):
		train(vid_format.format("train"), anno_format.format("train"), **vars(args)) #TODO
	elif(args["command"]=="metrics"):
		validate(vid_format.format(args["valid_set"]), anno_format.format(args["valid_set"]), **vars(args))

if __name__=="__main__":
	main()
