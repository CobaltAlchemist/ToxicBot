import torch
from ensemble import Ensemble
		

labels = ['toxic', 'severe_toxic', 'obscene', 'threat','insult','identity_hate']
ensemble = Ensemble('models')
torch.quantization.convert(ensemble, inplace=True)
	
def classify(s):
	pred = ensemble([s]) > 0.
	pred = pred.float().mean(0) > 0.5
	return pred.numpy()[0]
	
if __name__ == "__main__":
	print(classify('hello world'))
	print(classify('fuck you'))