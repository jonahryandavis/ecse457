import os
import json
from objdict import ObjDict
from itertools import izip
import argparse

xy_calc = 1.0
z_calc = 1.0

def loadJson(fname):
	with open(fname) as json_data:
		return json.load(json_data)

def grouped(iterable, n):
	return izip(*[iter(iterable)]*n)

def convertX(x):
	return round(x/xy_calc, 1)

def convertY(y):
	return round(y/xy_calc, 1)

def convertZ(z):
	return str(int(z/z_calc))

def addSegmentToJSON(jsonObj, segmentObj):
	if not segmentObj.has_key('materials'):
		return
	if not segmentObj.has_key('vertices'):
		return

	dbgName = segmentObj['materials'][0]['DbgName']
	for x, y, z in grouped(segmentObj['vertices'], 3):
		key = convertZ(z)
		if not jsonObj.has_key(key):
			jsonObj[key] = ObjDict()
		if not jsonObj[key].has_key(dbgName):
			jsonObj[key][dbgName] = []

		jsonObj[key][dbgName].append((convertX(x), convertY(y)))

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--d', type=str)
	parser.add_argument('--o', default='out.json', type=str)
	parser.add_argument('--xy_calc', type=float, default = 1.0)
	parser.add_argument('--z_calc', type=float, default = 1.0)
	args = parser.parse_args()

	xy_calc = args.xy_calc
	z_calc = args.z_calc
	
	jsonObj = ObjDict()
	for jsonFile in os.listdir(args.d):
		if jsonFile.endswith('.json'):
			print ('Parsing ' + args.d + jsonFile)
			segObj = loadJson(args.d + jsonFile)
			addSegmentToJSON(jsonObj, segObj)
	
	with open(args.o, 'w') as outFile:
		json.dump(jsonObj, outFile)
		outFile.close()
