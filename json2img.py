import os
import json
import argparse
from itertools import izip
import Image
import numpy as np
import math

WHITE = 255

def loadJson(fname):
	with open(fname) as json_data:
		return json.load(json_data)

def grouped(iterable, n):
	return izip(*[iter(iterable)]*n)

def boundaryArray(fnum, json_obj, x_scale, y_scale):
	boundary_array = [ [0] * x_scale for _ in range(y_scale)]
	for dbgName in json_obj:
		numVertices = len(json_obj[dbgName])
		for i in range(0, numVertices - 1):
			x0 = json_obj[dbgName][i][0]
			y0 = json_obj[dbgName][i][1]
			x1 = json_obj[dbgName][i+1][0]
			y1 = json_obj[dbgName][i+1][1]
			dist = math.sqrt(math.pow(x1-x0, 2) + math.pow(y1-y0, 2))
			if dist < math.sqrt(math.pow(x_scale, 2) + math.pow(y_scale, 2))/100:
				for j in range(0, int(round(dist))):
					boundary_array[int(round(y0 + j*(y1 - y0)/dist))][int(round(x0 + j*(x1 -x0)/dist))] = WHITE

	return boundary_array
	

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--j', type=str)
	parser.add_argument('--d', default='boundary', type=str)
	parser.add_argument('--o', default='boundary_', type=str)
	parser.add_argument('--x_scale', default=500, type=int)
	parser.add_argument('--y_scale', default=500, type=int)
	args = parser.parse_args()

	json_fname = args.j
	out_dir = args.d
	out_prefix = args.o
	x_scale = args.x_scale
	y_scale = args.y_scale

	json_data = loadJson(json_fname)

	if not os.path.exists(out_dir):
		os.makedirs(out_dir)

	for out_num in json_data:
		boundary_array = boundaryArray(out_num, json_data[out_num], x_scale, y_scale)
		im = Image.fromarray(np.asarray(boundary_array, dtype=np.uint8))
		im.save(str(out_dir) + '/' + str(out_prefix) + str(out_num) + '.jpg')

