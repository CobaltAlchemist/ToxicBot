import torch
import torch.nn as nn
import models
import os
import pickle
import glob
import json
import numpy as np

backend = 'fbgemm'
def split_path(path):
	_, path = os.path.splitdrive(path)
	folders = []
	while 1:
		path, folder = os.path.split(path)
		if folder != "":
			folders.append(folder)
		elif path == "\\" or path == "":
			break
	folders.reverse()
	return folders

if __name__ == "__main__":
	folder = 'models'
	for path in glob.glob(f'{folder}/*/'):
		model_name = split_path(path)[-1].split('_')[0]
		model_class = getattr(models, model_name)
		modelpath = os.path.join(path, 'model.pt')
		smallmodelpath = os.path.join(path, 'model_sml.pt')
		argspath = os.path.join(path, 'args.txt')
		
		with open(argspath, 'r') as f:
			args = json.loads(f.readline())
			
		model = model_class(**args)
		model.load_state_dict(torch.load(modelpath, map_location=torch.device('cpu')))
		model = model.half()
		model.eval()
		torch.save(model.state_dict(), smallmodelpath)