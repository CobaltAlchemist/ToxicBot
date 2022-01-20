import torch
import torch.nn as nn
import models
import os
import pickle
import glob
import json
import numpy as np

def collate_X(x, tokenizer, max_seq = 256):
	if max_seq == -1:
		x = [torch.Tensor(s.ids).long() for s in tokenizer.encode_batch(x)]
	else:
		x = [torch.Tensor(s.ids).long()[-max_seq:] for s in tokenizer.encode_batch(x)]
	lengths, idx = torch.Tensor([len(s) for s in x]).sort(0, descending=True)
	pseq = nn.utils.rnn.pad_sequence(x, batch_first=True)
	pseq = pseq[idx,:]
	return pseq, lengths

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

# Ensemble of modules
# Store model definitions in models.py
# Store models as:
#	folder
#	| - Class_<discriminator>
#	| - | - tokenizer.pkl
#	| - | - model.pt
#	| - | - args.txt (first line should include json dictionary args)
class Ensemble(nn.Module):
	def __init__(self, folder:str, model_limit = -1):
		super().__init__()
		self.ensemble = nn.ModuleList()
		self.tokenizers = []
		
		for folder in glob.glob(f'{folder}/*/'):
			model_name = split_path(folder)[-1].split('_')[0]
			model_class = getattr(models, model_name)
			modelpath = os.path.join(folder, 'model.pt')
			smallmodelpath = os.path.join(folder, 'model_sml.pt')
			argspath = os.path.join(folder, 'args.txt')
			tokenizerpath = os.path.join(folder, 'tokenizer.pkl')
			with open(argspath, 'r') as f:
				args = json.loads(f.readline())
			model = model_class(**args)
			try:
				model.load_state_dict(torch.load(smallmodelpath, map_location=torch.device('cpu')))
			except FileNotFoundError:
				model.load_state_dict(torch.load(modelpath, map_location=torch.device('cpu')))
			self.ensemble.append(model)
			with open(tokenizerpath, 'rb') as f:
				self.tokenizers.append(pickle.load(f))
			if len(self.ensemble) == model_limit:
				break
			
	def forward(self, s):
		preds = []
		for t, m in zip(self.tokenizers, self.ensemble):
			x = collate_X(s, t)
			x = (x[0].to(next(self.parameters()).device), x[1])
			with torch.no_grad():
				preds.append(m(x))
			x[0].cpu()
		preds = torch.stack(preds)
		return preds

if __name__ == "__main__":
	ensemble = Ensemble('models')
	print(ensemble(['hello world']))
	print(ensemble(['fuck you']))