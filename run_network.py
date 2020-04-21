#!/usr/bin/env python3

import os
import sys
import numpy as np
import math
import pickle
import argparse
import datetime
import deepdish as dd
import utils
import load_frames
import load_videos
import yolo_rnn_net
import mini
import network
import torch
import torchvision.transforms
from torch.utils.data import WeightedRandomSampler
from torch.utils.data import DataLoader
from torch.autograd import Variable
from collections import OrderedDict
from PIL import Image
from YoloLoss import YoloLoss

torch.multiprocessing.set_sharing_strategy("file_system") # prevents "0 items of ancdata" error

model = None
device = None
model_type = "" #TODO actually use this variable

anno_format = "/media/trocket/27276136-d5a4-4943-825f-7416775dc262/home/trocket/data/{}/annots/"
vid_format = "/media/trocket/27276136-d5a4-4943-825f-7416775dc262/home/trocket/data/{}/videos/"

# ### sample dataset
#anno_format = "sample_data/{}/annots/"
#vid_format = "sample_data/{}/videos/"

toTensor = torchvision.transforms.ToTensor()

def load_network(model_name, gpu, new, base_model=None): # gpu = 2 to use both, 0 and 1 for cuda:0 and cuda:1, -1 for cpu
	global model, device, model_type
	if("rnn" in model_name): # add additional when stuff is implemented
		model = yolo_rnn_net.YOLO_V1()
		model_type = "rnn"
	elif("lstm" in model_name):
		model = yolo_rnn_net.YOLO_V1(rnn_type="LSTM")
		model_type = "lstm"
	elif("tcnn" in model_name):
		model = yolo_rnn_net.YOLO_V1(rnn_type="TCNN")
		model_type= "tcnn"
	elif("mini" in model_name):
		model = mini.YOLO_V1()
		model_type = "mini"
	elif("darknet" in model_name):
		model = network.YOLO_V1()
		model.load_weight("../yolov1.weights")
		model_type = "darknet"
	else:
		model = network.YOLO_V1()
		model_type = "yolo"
	device = torch.device("cuda:"+str(0 if gpu==2 else gpu) if torch.cuda.is_available() and gpu!=-1 else "cpu")
	if(new):
		if(base_model!=None):
			state_dict = model.state_dict() # need to keep parameters not in base model
			for k, v in torch.load(base_model, map_location="cpu").items():
				if(k.startswith("module.")): k = k[7:] # in case model was saved with nn.DataParallel
				if(k in state_dict): state_dict[k] = v # only load parameters corresponding to new model
			model.load_state_dict(state_dict)
		print("Loaded trained model")
		model.train()
	else:
		state_dict = OrderedDict()
		for k, v in torch.load(model_name, map_location="cpu").items(): #in case model was saved with nn.DataParallel
			if(k.startswith("module.")): state_dict[k[7:]] = v
			else: state_dict[k] = v
		model.load_state_dict(state_dict)
		print("Loaded trained model")
		model.eval()
	if(torch.cuda.device_count() > 1 and gpu == 2):
		print("Let's use", torch.cuda.device_count(), "GPUs!")
		model = nn.DataParallel(model)
	model.to(device)

def run_interactive(vid_folder, anno_folder, **kwargs): #TODO add rnn support
	global model, device, model_type
	n_batch = 32 #kwargs["batch_size"]
	iou_thresh = kwargs["iou_thresh"]
	conf_thresh = kwargs["conf_thresh"]
	multiclass = kwargs["multiclass"]
	classdiff = kwargs["classdiff"]
	annotated = kwargs["annotated"]
	recurrent = model_type in ["rnn", "lstm", "tcnn"]
	while(True):
		try:
			v = input("Enter video number: ")
			anno = utils.load_anno(os.path.join(anno_folder, v+".pickle"))
			frames = [None]*len(anno)
			bboxes = []
			for i in range(len(anno)):
				frames[i] = toTensor(Image.open(os.path.join(vid_folder, "{}/{}.jpeg".format(v, i))).resize((224,224))).to(device)
			if(recurrent): hidden = None
			with(torch.no_grad()):
				for i in range(0, len(anno), n_batch):
					if(recurrent):
						imgs = torch.stack(frames[i:i+n_batch]).unsqueeze(0)
						yolout, hidden = model(imgs, h_prev=hidden, same_shape=True)
						bboxes += utils.yolo_to_bbox(yolout[0], conf_thresh=conf_thresh, iou_thresh=iou_thresh, multiclass=multiclass, classdiff=classdiff)
					else:
						imgs = torch.stack(frames[i:i+n_batch])
						yolout = model(imgs)
						bboxes += utils.yolo_to_bbox(yolout, conf_thresh=conf_thresh, iou_thresh=iou_thresh, multiclass=multiclass, classdiff=classdiff)
			if(annotated):
				utils.play_formatted_multi(os.path.join(vid_folder, v), 200, annotations=[bboxes, anno], class_labels=True)
			else:
				utils.play_formatted(os.path.join(vid_folder, v), 200, annotations=bboxes, class_labels=True)
		except KeyboardInterrupt:
			break
		#except: pass

def train(vid_folder, anno_folder, **kwargs): #TODO
	global model, device, model_type
	learning_rate = kwargs["learning_rate"]
	img_size = 224
	num_epochs = kwargs["epochs"]
	lambda_coord = kwargs["lambda_coord"]
	lambda_noobj = kwargs["lambda_noobj"]
	n_batch = kwargs["batch_size"]
	workers = kwargs["workers"]
	beta1 = kwargs["beta1"]
	beta2 = kwargs["beta2"]
	weight_decay = kwargs["weight_decay"]
	S = 7 # This is currently hardcoded into the YOLO model
	B = 2 # This is currently hardcoded into the YOLO model
	C = 24 # This is currently hardcoded into the YOLO model
	if(model_type in ["rnn", "lstm", "tcnn"]):
		recurrent = True
		train_dataset = load_videos.VideoDataset(vid_folder, anno_folder, 224, 7, 2, 24, [toTensor])
		y_train = np.array([i.data.tolist()[0][0] for i in train_dataset.labels])
		if(n_batch==-1): n_batch = 3
	else:
		recurrent = False
		train_dataset = load_frames.FramesDataset(vid_folder, anno_folder, 224, 7, 2, 24, [toTensor])
		y_train = np.array([i.data.tolist()[0] for i in train_dataset.labels])
		if(n_batch==-1): n_batch = 32
	class_sample_count = np.array([len(np.where(y_train == t)[0]) for t in range(C)])
	weight = [1/i if i > 0 else i for i in class_sample_count]
	samples_weight = np.array([weight[t] for t in y_train])
	samples_weight = torch.from_numpy(samples_weight)
	sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight), replacement=True)
	train_loader = DataLoader(train_dataset, batch_size=n_batch, num_workers=workers, sampler=sampler)
	loss_fn = YoloLoss(n_batch, B, C, lambda_coord, lambda_noobj, use_gpu=(device!=torch.device("cpu")), device=device)
	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(beta1, beta2))
	timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
	model_file = "results/model_{}_{}.pth".format(model_type, timestamp) #TODO add cli option for this
	loss_file = "results/loss_{}_{}.h5".format(model_type, timestamp) #TODO add cli option for this
	print("Model will be saved to "+model_file)
	print("Loss will be saved to "+loss_file)
	print("")
	loss_list = []
	for epoch in range(num_epochs): #TODO Should work for both normal and rnn, test it
		avgloss = 0
		for i,(data,target) in enumerate(train_loader):
			data = Variable(data).to(device)
			target = Variable(target).to(device)
			if(recurrent):
				pred, h = model(data)
			else:
				pred = model(data)
			loss = loss_fn(pred,target)
			current_loss = loss.data.cpu().numpy()
			avgloss += current_loss
			loss_list.append(current_loss)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			if i % 50 == 0:
				sys.stdout.write("\r%d/%d batches in %d/%d iteration, average loss was %f" % (i, len(train_loader), epoch+1, num_epochs, avgloss/50))
				sys.stdout.flush()
				avgloss = 0
				torch.save(model.state_dict(), model_file)
				dd.io.save(loss_file, np.array(loss_list))
	torch.save(model.state_dict(), model_file)
	print("Model saved successfully")
	dd.io.save(loss_file, np.array(loss_list))
	print("Loss saved successfully")


def validate(vid_folder, anno_folder, **kwargs): #TODO test
	global model, device, model_type
	n_batch = kwargs["batch_size"]
	workers = kwargs["workers"]
	iou_thresh = kwargs["iou_thresh"]
	conf_thresh = kwargs["conf_thresh"]
	multiclass = kwargs["multiclass"]
	classdiff = kwargs["classdiff"]
	results_file = kwargs["results_file"]
	if(results_file=="auto"):
		results_file = os.path.join("results/", "result_"+model_type+".pickle")
	if(model_type in ["rnn", "lstm", "tcnn"]):
		recurrent = True
		dataset = load_videos.VideoDataset(vid_folder, anno_folder, 224, 7, 2, 24, [toTensor], training=False)
		if(n_batch==-1): n_frames = 32
		else: n_frames = n_batch
		n_batch = 1
	else:
		recurrent = False
		dataset = load_frames.FramesDataset(vid_folder, anno_folder, 224, 7, 2, 24, [toTensor], training=False)
		if(n_batch==-1): n_batch = 32
	loader = DataLoader(dataset, batch_size=n_batch, num_workers=workers)
	preds = [[] for x in range(24)] # [[(confidence, iou)]]
	total_true = [0 for x in range(24)]
	for i, (data, batch_target) in enumerate(loader):
		#TODO torch turns batch_target into a weird list of 2 tensors, fix if you want to use multiple bounding boxes
		# the rest of the validation code supports multiple bboxes
		data = data.to(device)
		with torch.no_grad():
			if(recurrent):
				batch_boxes = []
				h = None
				for j in range(math.ceil(data.shape[1]/n_frames)):
					pred, h = model(data[:, j*n_frames:(j+1)*n_frames, :, :, :], h)
					batch_boxes += utils.yolo_to_bbox(pred, conf_thresh=conf_thresh, iou_thresh=iou_thresh, multiclass=multiclass, classdiff=classdiff)
			else:
				batch_boxes = utils.yolo_to_bbox(model(data), conf_thresh=conf_thresh, iou_thresh=iou_thresh, multiclass=multiclass, classdiff=classdiff)
		for b in range(len(batch_boxes)):
			if(recurrent):
				target = [(batch_target[b][0][0].item(), batch_target[b][1][0].tolist())]
				if(target[0][0]==-1): continue
			else:
				target = [(batch_target[0][b], batch_target[1][b,:].tolist())]
			bboxes = batch_boxes[b]
			used = [False]*len(target)
			for box in target: total_true[box[0]] += 1
			for bbox in bboxes:
				m = (-1,0)
				for j in range(len(target)):
					if(used[j] or bbox[0]!=target[j][0]): continue
					iou = utils.iou(bbox[1], target[j][1][0])
					if(iou>m[1]): m = (j, iou)
				if(m[0]!=-1):
					used[m[0]] = True
					preds[bbox[0]].append((bbox[2], m[1]))
				else: preds[bbox[0]].append((bbox[2], 0))
		if(i%20==0):
			sys.stdout.write("\rCompleted {}/{} batches".format(i, len(loader)))
			sys.stdout.flush()
	print("\nCompleted all inputs, calculating metrics...\n")
	results = {}
	ious = np.linspace(.5, .95, 10)
	for i in range(24):
		if(i==22): continue # Skip class None
		if(len(preds[i])==0): continue
		preds[i].sort(reverse=True)
		arr = np.array([b[1] for b in preds[i]])
		class_results = {}
		for iou in ious:
			pos_so_far = np.cumsum(arr>iou)
			precision = pos_so_far/np.array(range(1, arr.size+1))
			precision = np.maximum.accumulate(precision[::-1])[::-1]
			recall = pos_so_far/total_true[i]
			AP = np.sum(precision[1:]*np.diff(recall))+precision[0]*recall[0]
			iou_results = {}
			iou_results["AP"] = AP
			iou_results["Precision"] = precision[::int(precision.size/200)] if precision.size>200 else precision # store interpolation for plotting
			iou_results["Recall"] = recall[::int(recall.size/200)] if recall.size>200 else recall # store interpolation for plotting
			class_results["IOU "+str(iou)] = iou_results
		class_results["mAP"] = sum([item[1]["AP"] for item in class_results.items()])/10
		results[utils.classes[i]] = class_results
	results["mAP"] = sum([item[1]["mAP"] for item in results.items()])/23
	print(results)
	with open(results_file, 'wb+') as f:
		pickle.dump(results, f)


def main():
	parser = argparse.ArgumentParser(description="Run a model.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	subparsers = parser.add_subparsers(dest="command")

	# For anything required by all parsers
	parent_parser = argparse.ArgumentParser(add_help=False)
	parent_parser.add_argument("-m", "--model", required=True, help="model to run")
	parent_parser.add_argument("-g", "--gpu", default=0, type=int, help="which gpu to use")
	# TODO add --config so we can just use a config file

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
	train_parser = subparsers.add_parser("train",
		parents=[parent_parser, tm_parent_parser],
		help="train a model",
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	train_parser.add_argument("-b", "--base_model", default=None, help="model to load weights from")
	train_parser.add_argument("-lr", "--learning_rate", type=float, default=0.0006, help="learning rate for model training")
	train_parser.add_argument("-b1", "--beta1", type=float, default=0.9, help="beta1 for Adam optimizer")
	train_parser.add_argument("-b2", "--beta2", type=float, default=-1, help="beta2 for Adam optimizer, -1 for auto")
	train_parser.add_argument("-e", "--epochs", type=int, default=150, help="number of epochs to run, can be interrupted")
	train_parser.add_argument("-lc", "--lambda_coord", type=float, default=5, help="lambda_coord parameter for YOLO loss function")
	train_parser.add_argument("-ln", "--lambda_noobj", type=float, default=0.5, help="lambda_noobj parameter for YOLO loss function")
	train_parser.add_argument("-d", "--weight_decay", type=float, default=1e-4, help="weight decay for Adam optimizer")

	# For any arguments required by interactive
	interactive_parser = subparsers.add_parser("interactive",
		parents=[parent_parser, im_parent_parser],
		help="run a model on training videos interactively",
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	interactive_parser.add_argument("-a", "--annotated", action="store_true", help="Also show annotated boxes")

	# For any arguments required by metrics
	metrics_parser = subparsers.add_parser("metrics",
		parents=[parent_parser, im_parent_parser, tm_parent_parser],
		help="compute mAP and other metrics on a validation/test set",
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	metrics_parser.add_argument("-f", "--results_file", default="auto", help="file to write out result pickle to")

	args = vars(parser.parse_args())
	if(args["command"]==None):
		parser.print_help()
		exit(0)
	if("set" in args and args["set"] not in ["train", "test", "valid"]):
		print("Invalid choice of validation set, please choose train, valid, or test.")
		return
	if("beta2" in args and args["beta2"]==-1):
		args["beta2"] = 1-(1-args["beta1"])**2

	load_network(args["model"], args["gpu"], args["command"]=="train", args.get("base_model", None))
	if(args["command"]=="interactive"):
		run_interactive(vid_format.format(args["set"]), anno_format.format(args["set"]), **args)
	elif(args["command"]=="train"):
		train(vid_format.format("train"), anno_format.format("train"), **args) #TODO
	elif(args["command"]=="metrics"):
		validate(vid_format.format(args["set"]), anno_format.format(args["set"]), **args)

if __name__=="__main__":
	main()
