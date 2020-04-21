#!/usr/bin/env python3

import pickle
import sys

if __name__ == "__main__":
	with open(sys.argv[1], 'rb') as f:
		r = pickle.load(f)
	a = 0
	for k, v in r.items():
		if(k!="mAP"):
			print(k, v["IOU 0.5"]["AP"])
			a += v["IOU 0.5"]["AP"]
	print("mAP", a/22)
