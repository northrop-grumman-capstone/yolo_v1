import os
import cv2
import numpy as np
import pickle
import torch

classes = ["Person", "Bird", "Bicycle", "Boat", "Bus", "Bear", "Cow", "Cat", "Giraffe",
           "Potted Plant", "Horse", "Motorcycle", "Knife", "Airplane", "Skateboard",
           "Train", "Truck", "Zebra", "Toilet", "Dog", "Elephant", "Umbrella", "None", "Car"]

anno_format = "/media/trocket/27276136-d5a4-4943-825f-7416775dc262/home/trocket/data/{}/annots/{}.pickle"
vid_format = "/media/trocket/27276136-d5a4-4943-825f-7416775dc262/home/trocket/data/{}/videos/{}"

def bbox_to_rect(bbox, width, height): #bbox is [xmin, xmax, ymin, ymax] as floats between 0 and 1
	return (int(bbox[0]*width), int(bbox[2]*height), int((bbox[1]-bbox[0])*width), int((bbox[3]-bbox[2])*height))

def rect_to_bbox(rect, width, height): #rect is (xmin, ymin, width, height) as ints in img coords
	return [rect[0]/width, (rect[0]+rect[2])/width, rect[1]/height, (rect[1]+rect[3])/height]

def iou(box1, box2): # boxes are np array [xmin, xmax, ymin, ymax]
	wi = min(box1[1], box2[1])-max(box1[0], box2[0])
	hi = min(box1[3], box2[3])-max(box1[2], box2[2])
	if(wi<0 or hi<0): return 0
	inter = wi*hi
	union = (box1[1]-box1[0])*(box1[3]-box1[2])+(box2[1]-box2[0])*(box2[3]-box2[2])-inter
	return inter/union

def nonmaxsuppress(l, iou_thresh, class_diff): # l = [(class, [xmin, xmax, ymin, ymax], confidence)]
	l.sort(key=lambda x: x[2], reverse=True) # sort by confidence
	boxes = []
	for box1 in l:
		a = True
		for box2 in boxes:
			if((not class_diff or box1[0]==box2[0]) and iou(box1[1],box2[1])>iou_thresh):
				a = False
				break
		if(a):
			boxes.append(box1)
	return boxes

def yolo_to_bbox(yolout, conf_thresh=0.25, iou_thresh=0.3, multiclass=False, classdiff=True, S=7, B=2, C=24): # converts yolo output [batch, S*S*(B*5+C)] to [[(class, [xmin, xmax, ymin, ymax])]]
	if(len(yolout.size())==1): yolout.unsqueeze(0)
	assert classdiff if multiclass else True, "Please don't use multiclass without classdiff."
	batch_boxes = []
	n_elements = B*5+C
	for b in range(yolout.size()[0]):
		boxes = []
		for i in range(S):
			for j in range(S):
				x = (S*i+j)*n_elements
				box_data = yolout[b][x:x+B*5].tolist()
				classes = yolout[b][x+B*5:x+n_elements]
				for box in range(B):
					if(multiclass):
						pred_classes = ((box_data[box*5+4]*classes)>conf_thresh).nonzero().flatten().tolist()
						if(len(pred_classes)==0): continue
					else:
						pred_classes = [classes.argmax().item()]
						confidence = box_data[box*5+4]*classes[pred_classes[0]]
						if(confidence<conf_thresh): continue
					# xc and yc are offsets from grid, change back to box_data[0] and [1] if works poorly
					#xc = (i+box_data[0])/S
					#yc = (j+box_data[1])/S
					xc = box_data[0]
					yc = box_data[1]
					w = box_data[2]**2
					h = box_data[3]**2
					arr = np.array([xc-w/2, xc+w/2, yc-h/2, yc+h/2])
					for pred_class in pred_classes:
						confidence = box_data[box*5+4]*classes[pred_class]
						boxes.append((pred_class, arr, confidence))
		boxes = nonmaxsuppress(boxes, iou_thresh, classdiff)
		batch_boxes.append(boxes)
	#print(batch_boxes)
	return batch_boxes

def draw_bbox(img, xy1, xy2, color=(255,0,0), label="", t_color=(0,0,0), thickness=4, t_scale=1, t_thickness=2):
	color = (color[2], color[1], color[0])
	t_color = (t_color[2], t_color[1], t_color[0])
	bbox_img = img.copy()
	cv2.rectangle(bbox_img, xy1, xy2, color, thickness=thickness)
	if(label!=""):
		box, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, t_scale, t_thickness)
		hb = int(baseline/2)
		box = (box[0]+baseline, box[1]+baseline)
		cv2.rectangle(bbox_img, (xy1[0],xy1[1]), (xy1[0]+box[0],xy1[1]+box[1]), color, -1)
		cv2.putText(bbox_img, label, (xy1[0]+hb,xy1[1]+box[1]-hb), cv2.FONT_HERSHEY_SIMPLEX, t_scale, t_color, t_thickness) #TODO ????
	return bbox_img

def show(img, title="image", time=0, destroy=True):
	cv2.imshow(title, img)
	k = cv2.waitKey(time)
	if(destroy): cv2.destroyAllWindows()
	return k

def load_anno(filename):
	with open(filename, 'rb') as f:
		return pickle.load(f)

def play_video(filename, frame_time, title="video", annotations=None, class_labels=False):
	video = cv2.VideoCapture(filename)
	frame_width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
	frame_height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
	if(not video.isOpened()):
		print("Failed to open video")
	while(True):
		ok, frame = video.read()
		frame_num = video.get(cv2.CAP_PROP_POS_FRAMES)-1
		if(not ok): break
		if(annotations != None and len(annotations)>frame_num): #TODO add annotation to frame
			#TODO different color for each bbox
			for anno in annotations[int(frame_num)]:
				xy1 = (int(anno[1][0]*frame_width), int(anno[1][2]*frame_height))
				xy2 = (int(anno[1][1]*frame_width), int(anno[1][3]*frame_height))
				if(class_labels):
					frame = draw_bbox(frame, xy1, xy2, label=classes[anno[0]])
				else:
					frame = draw_bbox(frame, xy1, xy2)
		cv2.imshow(title, frame)
		k = cv2.waitKey(frame_time)
		if(k == 97): # go back one frame when 'a' is pressed
			video.set(cv2.CAP_PROP_POS_FRAMES, frame_num-1 if frame_num>0 else 0)
		if(k == 120): # exit video when 'x' is pressed (forward otherwise)
			break
	try: cv2.destroyWindow(title)
	except: pass

def play_formatted(foldername, frame_time, title="video", annotations=None, class_labels=False, out_size=None):
	frame_num = 0
	while(True):
		try:
			#frame = np.load("{}/{}.npy".format(foldername, frame_num))
			frame = cv2.imread("{}/{}.jpeg".format(foldername, frame_num))
			if(out_size!=None):
				frame = cv2.resize(frame, out_size)
			frame_height = frame.shape[0]
			frame_width = frame.shape[1]
			if(annotations != None and len(annotations)>frame_num): #TODO add annotation to frame
				#TODO different color for each bbox
				for anno in annotations[int(frame_num)]:
					xy1 = (int(anno[1][0]*frame_width), int(anno[1][2]*frame_height))
					xy2 = (int(anno[1][1]*frame_width), int(anno[1][3]*frame_height))
					if(class_labels):
						frame = draw_bbox(frame, xy1, xy2, label=classes[anno[0]])
					else:
						frame = draw_bbox(frame, xy1, xy2)
			cv2.imshow(title, frame)
			k = cv2.waitKey(frame_time)
			if(k == 97): # go back one frame when 'a' is pressed
				frame_num = frame_num-1 if frame_num>0 else 0
			if(k == 120): # exit video when 'x' is pressed (forward otherwise)
				break
			else:
				frame_num += 1
		except: break
	try: cv2.destroyWindow(title)
	except: pass

def play_formatted_multi(foldername, frame_time, title="video", annotations=None, class_labels=False, out_size=None):
	frame_num = 0
	colors = [(255,0,0), (0,255,0), (0,0,255), (128,0,128), (0,128,128), (128,128,0), (128,128,128)]
	while(True):
		try:
			#frame = np.load("{}/{}.npy".format(foldername, frame_num))
			frame = cv2.imread("{}/{}.jpeg".format(foldername, frame_num))
			if(out_size!=None):
				frame = cv2.resize(frame, out_size)
			frame_height = frame.shape[0]
			frame_width = frame.shape[1]
			if(annotations != None and len(annotations[0])>frame_num): #TODO add annotation to frame
				for i in range(len(annotations)):
					for anno in annotations[i][int(frame_num)]:
						xy1 = (int(anno[1][0]*frame_width), int(anno[1][2]*frame_height))
						xy2 = (int(anno[1][1]*frame_width), int(anno[1][3]*frame_height))
						if(class_labels):
							frame = draw_bbox(frame, xy1, xy2, label=classes[anno[0]], color=colors[i])
						else:
							frame = draw_bbox(frame, xy1, xy2, color=colors[i])
			cv2.imshow(title, frame)
			k = cv2.waitKey(frame_time)
			if(k == 97): # go back one frame when 'a' is pressed
				frame_num = frame_num-1 if frame_num>0 else 0
			if(k == 120): # exit video when 'x' is pressed (forward otherwise)
				break
			else:
				frame_num += 1
		except: break
	try: cv2.destroyWindow(title)
	except: pass

def play(s,n):
	a = load_anno(anno_format.format(s,n))
	play_formatted(vid_format.format(s,n), 100, annotations=a, class_labels=True, out_size=(500,500))
