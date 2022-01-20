import numpy as np # linear algebra
import torch
import torch.nn as nn
import torch.nn.functional as F
# [CNN, GRU, SelfAttention, XformEncoder, XformGru, GruAttention, HydraGruAttention, ResnetHydraGruAttn]
models = [
	'CNN',
	'GRU',
	'SelfAttention',
	'XformEncoder',
	'XformGru',
	'GruAttention',
	'HydraGruAttention',
	'ResnetHydraGruAttn',
]

class CNN(nn.Module):
	def __init__(self, emb_dims = 512, hidden:int=128, layers:int=2, classes:int=6, **kwargs):
		super().__init__()
		self.embeddings = nn.Embedding(20_000, emb_dims)
		self.CNN = nn.Sequential(
			nn.Conv1d(emb_dims, hidden, 3),
			nn.Dropout(0.3),
			nn.ReLU(),
			nn.Conv1d(hidden, hidden, 3),
			nn.Dropout(0.3),
			nn.ReLU(),
			nn.AdaptiveMaxPool1d(1),
		)
		self.linear = nn.Sequential(
			nn.Dropout(0.3),
			nn.Linear(hidden, hidden),
			nn.ReLU(),
			nn.Dropout(0.3),
			nn.Linear(hidden, classes),
		)

	def forward(self, x, lengths=None):
		if lengths is None:
			x, lengths = x
		x = self.embeddings(x)
		x = torch.transpose(x, 1, 2)
		x = self.CNN(x)
		return self.linear(x[...,0])

class GRU(nn.Module):
	def __init__(self, emb_dims = 512, hidden:int=128, layers:int=1, classes:int=6, **kwargs):
		super().__init__()
		self.embeddings = nn.Embedding(20_000, emb_dims)
		self.GRU = nn.GRU(
			input_size = emb_dims,
			hidden_size = hidden,
			num_layers = layers,
			bidirectional=True,
			dropout=0.3,
			batch_first=True,
		)
		self.linear = nn.Sequential(
			nn.Dropout(0.3),
			nn.Linear(hidden*5, hidden),
			nn.ReLU(),
			nn.Dropout(0.3),
			nn.Linear(hidden, classes),
		)

	def forward(self, x, lengths=None):
		if lengths is None:
			x, lengths = x
		x = self.embeddings(x)
		x = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True)
		x, h = self.GRU(x)
		x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
		avg = x.mean(dim=1)
		max = x.max(dim=1)[0]
		x = torch.cat((h[-1], avg, max), dim=1)
		#x = F.relu(x)
		return self.linear(x)
		
class SelfAttention(nn.Module):
	def __init__(self, emb_dims = 512, hidden:int=256, heads=8, layers:int=2, classes:int=6, **kwargs):
		super().__init__()
		self.embeddings = nn.Embedding(20_000, emb_dims)
		self.attn = nn.MultiheadAttention(
			embed_dim=emb_dims,
			num_heads=heads,
		)
		self.norm = nn.LayerNorm(emb_dims)
		self.linear = nn.Sequential(
			nn.Dropout(0.3),
			nn.Linear(emb_dims, hidden),
			nn.ReLU(),
			nn.Dropout(0.3),
			nn.Linear(hidden, classes),
		)

	def forward(self, x, lengths=None):
		if lengths is None:
			x, lengths = x
		x = self.embeddings(x)
		x = torch.transpose(x, 0, 1)
		attn, _ = self.attn(x, x, x)
		x = self.norm(x + attn)
		x = x.mean(0)
		x = self.linear(x)
		return x
		
class XformEncoder(nn.Module):
	def __init__(self, emb_dims = 512, hidden:int=1024, heads=8, layers:int=1, classes:int=6, **kwargs):
		super().__init__()
		self.embeddings = nn.Embedding(20_000, emb_dims)
		self.encoder_layer = nn.TransformerEncoderLayer(
			d_model=emb_dims,
			nhead=heads,
			dim_feedforward=hidden
		)
		self.encoder = nn.TransformerEncoder(
			encoder_layer=self.encoder_layer,
			num_layers=layers,
		)
		self.linear = nn.Linear(2*emb_dims, classes)

	def forward(self, x, lengths=None):
		if lengths is None:
			x, lengths = x
		x = self.embeddings(x)
		x = torch.transpose(x, 0, 1)
		x = self.encoder(x)
		x = torch.cat((x.mean(0), x.max(0)[0]), dim=1)
		x = self.linear(x)
		return x
		
class XformGru(nn.Module):
	def __init__(self, emb_dims = 256, hidden:int=256, heads=8, layers:int=1, classes:int=6, **kwargs):
		super().__init__()
		self.embeddings = nn.Embedding(20_000, emb_dims)
		self.encoder_layer = nn.TransformerEncoderLayer(
			d_model=emb_dims,
			nhead=heads,
			dim_feedforward=hidden
		)
		self.encoder = nn.TransformerEncoder(
			encoder_layer=self.encoder_layer,
			num_layers=layers,
		)
		self.GRU = nn.GRU(
			input_size = emb_dims,
			hidden_size = hidden,
			num_layers = layers,
			bidirectional=True,
			dropout=0.3,
			batch_first=True,
		)
		self.linear = nn.Sequential(
			nn.Dropout(0.3),
			nn.Linear(hidden*5 + 2*emb_dims, hidden),
			nn.ReLU(),
			nn.Dropout(0.3),
			nn.Linear(hidden, classes),
		)

	def forward(self, x, lengths=None):
		if lengths is None:
			x, lengths = x
		x = self.embeddings(x)
		
		#Attention
		attn = torch.transpose(x, 0, 1)
		attn = self.encoder(attn)
		attn = torch.cat((attn.mean(0), attn.max(0)[0]), dim=1)
		
		#Gru
		gru = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True)
		gru, gru_h = self.GRU(gru)
		gru, _ = nn.utils.rnn.pad_packed_sequence(gru, batch_first=True)
		gru = torch.cat((gru_h[-1], gru.mean(dim=1), gru.max(dim=1)[0]), dim=1)
		
		x = torch.cat((attn, gru), dim=1)
		x = self.linear(x)
		return x
		
class GruAttention(nn.Module):
	def __init__(self, emb_dims = 512, hidden:int=128, heads=8, layers:int=1, vocab_size=20_000, dropout=0.3, classes:int=6, **kwargs):
		super().__init__()
		self.embeddings = nn.Embedding(vocab_size, emb_dims)
		self.GRU = nn.GRU(
			input_size = emb_dims,
			hidden_size = hidden,
			num_layers = layers,
			bidirectional=True,
			dropout=dropout,
			batch_first=True,
		)
		self.attn = nn.MultiheadAttention(
			embed_dim=hidden*2,
			num_heads=heads,
		)
		self.linear = nn.Sequential(
			nn.Dropout(dropout),
			nn.Linear(hidden*4, hidden),
			nn.ReLU(),
			nn.Dropout(dropout),
			nn.Linear(hidden, classes),
		)

	def forward(self, x, lengths=None):
		if lengths is None:
			x, lengths = x
		x = self.embeddings(x)
		
		#Gru
		x = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True)
		x, _ = self.GRU(x)
		x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=False)
		
		#Attention
		x, _ = self.attn(x,x,x)
		x = torch.cat((x.mean(0), x.max(0)[0]), dim=1)
		
		x = self.linear(x)
		return x
		
class HydraGruAttention(nn.Module):
	def __init__(self, emb_dims = 512, hidden:int=128, heads=8, layers:int=1, vocab_size=20_000, dropout=0.3, classes:int=6, **kwargs):
		super().__init__()
		self.embeddings = nn.Embedding(vocab_size, emb_dims)
		self.GRU = nn.GRU(
			input_size = emb_dims,
			hidden_size = hidden//2,
			num_layers = layers,
			bidirectional=True,
			dropout=dropout,
			batch_first=True,
		)
		self.attn = nn.MultiheadAttention(
			embed_dim=hidden,
			num_heads=heads,
		)
		self.linear = nn.ModuleList([nn.Sequential(
			nn.Dropout(dropout),
			nn.Linear(hidden*2, hidden),
			nn.ReLU(),
			nn.Dropout(dropout),
			nn.Linear(hidden, 1)
		) for _ in range(classes)])

	def forward(self, x, lengths=None):
		if lengths is None:
			x, lengths = x
		x = self.embeddings(x)
		
		#Gru
		x = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True)
		x, _ = self.GRU(x)
		x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=False)
		
		#Attention
		x, _ = self.attn(x,x,x)
		x = torch.cat((x.mean(0), x.max(0)[0]), dim=1)
		
		x = torch.cat([l(x) for l in self.linear], dim=1)
		return x
		
class AttnBlock(nn.Module):
	def __init__(self, emb_dims:int=2048, kdim:int=None, vdim:int=None, heads:int=8):
		super().__init__()
		self.act1 = nn.ReLU()
		self.attn1 = nn.MultiheadAttention(embed_dim=emb_dims, num_heads=heads, kdim=kdim, vdim=vdim)
		self.act2 = nn.ReLU()
		self.attn2 = nn.MultiheadAttention(embed_dim=emb_dims, num_heads=heads, kdim=kdim, vdim=vdim)
		
	def forward(self, x):
		skip = x
		x = self.act1(x)
		x, _ = self.attn1(x, x, x)
		x = self.act2(x)
		x, _ = self.attn2(x, x, x)
		return x + skip

class ResnetAttention(nn.Module):
	def __init__(self, emb_dims = 512, hidden:int=512, heads=8, layers:int=16, vocab_size=20_000, dropout=0.3, classes:int=6, **kwargs):
		super().__init__()
		self.embeddings = nn.Embedding(vocab_size, emb_dims)
		self.attn = nn.Sequential(
			*(AttnBlock(emb_dims, heads=heads) for _ in range(layers)),
			nn.ReLU(),
		)
		self.linear = nn.Sequential(
			nn.Dropout(dropout),
			nn.Linear(emb_dims*2, hidden),
			nn.ReLU(),
			nn.Dropout(dropout),
			nn.Linear(hidden, classes),
		)

	def forward(self, x, lengths=None):
		if lengths is None:
			x, lengths = x
		x = self.embeddings(x)
		x = self.attn(x)
		x = torch.cat((x.mean(1), x.max(1)[0]), dim=1)
		x = self.linear(x)
		return x
		
class GruAttnBlock(nn.Module):
	def __init__(self, emb_dims:int=512, hidden:int=128, kdim:int=None, vdim:int=None, heads:int=8, dropout:float=0.3, **kwargs):
		super().__init__()
		self.GRU = nn.GRU(
			input_size = emb_dims,
			hidden_size = hidden//2,
			num_layers = 1,
			bidirectional=True,
			dropout=dropout,
			batch_first=True,
		)
		self.attn = nn.MultiheadAttention(embed_dim=hidden, num_heads=heads, kdim=kdim, vdim=vdim)
		self.act = nn.ReLU()
		
	def forward(self, x, lengths=None):
		if lengths is None:
			x, lengths = x
		#Gru
		x = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True)
		x, _ = self.GRU(x)
		x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=False)
		#Attn
		x, _ = self.attn(x,x,x)
		x = self.act(x)
		x = torch.transpose(x, 0, 1) # batch first
		return (x, lengths)

class ResnetHydraGruAttn(nn.Module):
	def __init__(self, emb_dims = 512, hidden:int=128, heads=8, layers:int=2, vocab_size=20_000, dropout=0.3, classes:int=6, **kwargs):
		super().__init__()
		self.embeddings = nn.Embedding(vocab_size, emb_dims)
		self.blocks = nn.Sequential(
			GruAttnBlock(emb_dims=emb_dims, hidden=hidden, heads=heads, dropout=dropout),
			*(GruAttnBlock(emb_dims=hidden, hidden=hidden, heads=heads, dropout=dropout) for _ in range(layers-1)),
		)
		self.linear = nn.ModuleList([nn.Sequential(
			nn.Dropout(dropout),
			nn.Linear(hidden*2, hidden),
			nn.ReLU(),
			nn.Dropout(dropout),
			nn.Linear(hidden, 1)
		) for _ in range(classes)])

	def forward(self, x, lengths=None):
		if lengths is None:
			x, lengths = x
		x = self.embeddings(x)
		x, lengths = self.blocks((x, lengths))
		x = torch.cat((x.mean(1), x.max(1)[0]), dim=1)
		x = torch.cat([l(x) for l in self.linear], dim=1)
		return x
		
class DeepHydraGruAttnHead(nn.Module):
	def __init__(self, emb_dims = 512, hidden:int=128, heads=8, layers:int=1, dropout=0.3, **kwargs):
		super().__init__()
		self.GRU = nn.GRU(
			input_size = emb_dims,
			hidden_size = hidden,
			num_layers = layers,
			bidirectional=True,
			dropout=dropout,
			batch_first=True,
		)
		self.attn = nn.MultiheadAttention(
			embed_dim=hidden*2,
			num_heads=heads,
		)
		self.linear = nn.Sequential(
			nn.Dropout(dropout),
			nn.Linear(hidden*4, hidden),
			nn.ReLU(),
			nn.Dropout(dropout),
			nn.Linear(hidden, 1),
		)

	def forward(self, x, lengths=None):
		if lengths is None:
			x, lengths = x
		
		#Gru
		x = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True)
		x, _ = self.GRU(x)
		x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=False)
		
		#Attention
		x, _ = self.attn(x,x,x)
		x = torch.cat((x.mean(0), x.max(0)[0]), dim=1)
		
		x = self.linear(x)
		return x
		
class DeepHydraGruAttn(nn.Module):
	def __init__(self, emb_dims = 512, hidden:int=128, heads=8, layers:int=2, vocab_size=20_000, dropout=0.3, classes:int=6, **kwargs):
		super().__init__()
		self.embeddings = nn.Embedding(vocab_size, emb_dims)
		
		self.GRU = nn.GRU(
			input_size = emb_dims,
			hidden_size = hidden//2,
			num_layers = layers,
			bidirectional=True,
			dropout=dropout,
			batch_first=True,
		)
		self.attn = nn.MultiheadAttention(
			embed_dim=hidden,
			num_heads=heads,
		)
		
		self.heads = nn.ModuleList([DeepHydraGruAttnHead(
			emb_dims=hidden, hidden=hidden, heads=heads, layers=layers, dropout=dropout
		) for _ in range(classes)])
	
	def forward(self, x, lengths=None):
		if lengths is None:
			x, lengths = x
		lengths = lengths.cpu()
		x = self.embeddings(x)
		
		#Gru
		x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True)
		x, _ = self.GRU(x)
		x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=False)
		
		#Attention
		x, _ = self.attn(x,x,x)
		x = torch.transpose(x, 0, 1)
		
		x = torch.cat([h(x, lengths) for h in self.heads], dim=1)
		return x
		
class ToxicOnly(nn.Module):
	def __init__(self, *args, **kwargs):
		super().__init__()
		self.model = ResnetHydraGruAttn(*args, **kwargs)
		
	def forward(self, x, lengths=None):
		x = self.model(x, lengths)
		x[1:] = 0
		return x
	
	
class PureHydraGruAttn(nn.Module):
	def __init__(self, emb_dims = 512, hidden:int=128, heads=8, layers:int=2, vocab_size=20_000, dropout=0.3, classes:int=6, **kwargs):
		super().__init__()
		self.embeddings = nn.Embedding(vocab_size, emb_dims)
	
		self.heads = nn.ModuleList([DeepHydraGruAttnHead(
			emb_dims=emb_dims, hidden=hidden, heads=heads, layers=layers, dropout=dropout
		) for _ in range(classes)])
	
	def forward(self, x, lengths=None):
		if lengths is None:
			x, lengths = x
		lengths = lengths.cpu()
		x = self.embeddings(x)
		
		x = torch.cat([h(x, lengths) for h in self.heads], dim=1)
		return x
	
class CNNGruAttn(nn.Module):
	def __init__(self, emb_dims = 512, hidden:int=128, heads=8, layers:int=2, vocab_size=20_000, dropout=0.3, classes:int=6, **kwargs):
		super().__init__()
		self.embeddings = nn.Embedding(vocab_size, emb_dims)
		
		self.CNN = nn.Sequential(
			nn.Conv1d(emb_dims, hidden, 3, padding=1),
			nn.Dropout(dropout),
			nn.ReLU(),
		)
		
		self.GRU = nn.GRU(
			input_size = hidden,
			hidden_size = hidden//2,
			num_layers = layers,
			bidirectional=True,
			dropout=dropout,
			batch_first=True,
		)
		
		self.attn = nn.MultiheadAttention(
			embed_dim=hidden,
			num_heads=heads,
		)
		
		self.linear = nn.ModuleList([nn.Sequential(
			nn.Dropout(dropout),
			nn.Linear(hidden*2, hidden),
			nn.ReLU(),
			nn.Dropout(dropout),
			nn.Linear(hidden, 1)
		) for _ in range(classes)])
	
	def forward(self, x, lengths=None):
		if lengths is None:
			x, lengths = x
		lengths = lengths.cpu()
		x = self.embeddings(x)
		
		#CNN
		x = torch.transpose(x, 1, 2)
		x = self.CNN(x)
		x = torch.transpose(x, 1, 2)
		
		#Gru
		x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True)
		x, _ = self.GRU(x)
		x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=False)
		
		#Attention
		x, _ = self.attn(x,x,x)
		x = torch.cat((x.mean(0), x.max(0)[0]), dim=1)
		
		x = torch.cat([l(x) for l in self.linear], dim=1)
		return x
	
class CNNGruAttnSplit(nn.Module):
	def __init__(self, emb_dims = 512, hidden:int=128, heads=8, layers:int=2, vocab_size=20_000, dropout=0.3, classes:int=6, **kwargs):
		super().__init__()
		self.embeddings = nn.Embedding(vocab_size, emb_dims)
		
		self.CNN = nn.Sequential(
			nn.Conv1d(emb_dims, hidden, 3, padding=1),
			nn.Dropout(dropout),
			nn.ReLU(),
		)
		
		self.GRU = nn.GRU(
			input_size = emb_dims,
			hidden_size = hidden//2,
			num_layers = layers,
			bidirectional=True,
			dropout=dropout,
			batch_first=True,
		)
		
		self.attn = nn.MultiheadAttention(
			embed_dim=emb_dims,
			num_heads=heads,
		)
		
		self.linear = nn.ModuleList([nn.Sequential(
			nn.Dropout(dropout),
			nn.Linear(hidden*4+emb_dims*2, hidden),
			nn.ReLU(),
			nn.Dropout(dropout),
			nn.Linear(hidden, 1)
		) for _ in range(classes)])
	
	def forward(self, x, lengths=None):
		if lengths is None:
			x, lengths = x
		lengths = lengths.cpu()
		x = self.embeddings(x)
		
		#CNN
		cnn_o = torch.transpose(x, 1, 2)
		cnn_o = self.CNN(cnn_o) #(B, C, N)
		
		#Gru
		gru_o = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True)
		gru_o, _ = self.GRU(gru_o)
		gru_o, _ = nn.utils.rnn.pad_packed_sequence(gru_o, batch_first=False) #(N, B, C)
		
		#Attention
		attn_o = torch.transpose(x, 0, 1)
		attn_o, _ = self.attn(attn_o,attn_o,attn_o) #(N, B, C)
		
		x = torch.cat((
			cnn_o.mean(-1),
			cnn_o.max(-1)[0],
			gru_o.mean(0),
			gru_o.max(0)[0],
			attn_o.mean(0),
			attn_o.max(0)[0],
		), dim=1)
		
		x = torch.cat([l(x) for l in self.linear], dim=1)
		return x
	
class CNNGruAttnCascade(nn.Module):
	def __init__(self, emb_dims = 512, hidden:int=128, heads=8, layers:int=2, vocab_size=20_000, dropout=0.3, classes:int=6, **kwargs):
		super().__init__()
		self.embeddings = nn.Embedding(vocab_size, emb_dims)
		
		self.CNN = nn.Sequential(
			nn.Conv1d(emb_dims, hidden, 3, padding=1),
			nn.Dropout(dropout),
			nn.ReLU(),
		)
		
		self.GRU = nn.GRU(
			input_size = hidden+emb_dims,
			hidden_size = hidden//2,
			num_layers = layers,
			bidirectional=True,
			dropout=dropout,
			batch_first=True,
		)
		
		self.attn = nn.MultiheadAttention(
			embed_dim=hidden*2+emb_dims,
			num_heads=heads,
		)
		
		self.linear = nn.ModuleList([nn.Sequential(
			nn.Dropout(dropout),
			nn.Linear((hidden*2+emb_dims)*4, hidden),
			nn.ReLU(),
			nn.Dropout(dropout),
			nn.Linear(hidden, 1)
		) for _ in range(classes)])
	
	def forward(self, x, lengths=None):
		if lengths is None:
			x, lengths = x
		lengths = lengths.cpu()
		x = self.embeddings(x)
		
		#CNN
		cnn = torch.transpose(x, 1, 2)
		cnn = self.CNN(cnn)
		cnn = torch.transpose(cnn, 1, 2)
		x = torch.cat((x, cnn), dim=2)
		
		#Gru
		gru = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True)
		gru, _ = self.GRU(gru)
		gru, _ = nn.utils.rnn.pad_packed_sequence(gru, batch_first=False)
		x = torch.transpose(x, 0, 1)
		x = torch.cat((x, gru), dim=2)
		
		#Attention
		attn, _ = self.attn(x,x,x)
		x = torch.cat((x, attn), dim=2)
		
		x = torch.cat((x.mean(0), x.max(0)[0]), dim=1)
		
		x = torch.cat([l(x) for l in self.linear], dim=1)
		return x
	
class FFTConv1d(nn.Conv1d):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
	
	def _conv_forward(self, input, weight, bias):
		input = F.pad(input, self.padding + self.padding)
		padded_weight = F.pad(weight, (0, input.shape[-1] - weight.shape[-1]))
		input_ft = torch.fft.rfft(input)
		weight_ft = torch.fft.rfft(padded_weight)
		
		out_ft = torch.einsum("ab..., cb... -> ac...", input_ft, weight_ft)
		out = torch.fft.irfft(out_ft)
		if bias is not None:
			out = out + bias[None, :, None]
		return out
		
class FFTCNNGruAttn(nn.Module):
	def __init__(self, emb_dims = 512, hidden:int=128, heads=8, layers:int=2, vocab_size=20_000, dropout=0.3, classes:int=6, **kwargs):
		super().__init__()
		self.embeddings = nn.Embedding(vocab_size, emb_dims)
		
		self.CNN = nn.Sequential(
			FFTConv1d(emb_dims, hidden, 64, padding=1),
			nn.Dropout(dropout),
			nn.ReLU(),
		)
		
		self.GRU = nn.GRU(
			input_size = hidden,
			hidden_size = hidden//2,
			num_layers = layers,
			bidirectional=True,
			dropout=dropout,
			batch_first=True,
		)
		
		self.attn = nn.MultiheadAttention(
			embed_dim=hidden,
			num_heads=heads,
		)
		
		self.linear = nn.ModuleList([nn.Sequential(
			nn.Dropout(dropout),
			nn.Linear(hidden*2, hidden),
			nn.ReLU(),
			nn.Dropout(dropout),
			nn.Linear(hidden, 1)
		) for _ in range(classes)])
	
	def forward(self, x, lengths=None):
		if lengths is None:
			x, lengths = x
		lengths = lengths.cpu()
		x = self.embeddings(x)
		
		#CNN
		x = torch.transpose(x, 1, 2)
		x = self.CNN(x)
		x = torch.transpose(x, 1, 2)
		
		#Gru
		x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True)
		x, _ = self.GRU(x)
		x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=False)
		
		#Attention
		x, _ = self.attn(x,x,x)
		x = torch.cat((x.mean(0), x.max(0)[0]), dim=1)
		
		x = torch.cat([l(x) for l in self.linear], dim=1)
		return x


class FFTCNN(nn.Module):
	def __init__(self, emb_dims = 512, hidden:int=128, layers:int=2, classes:int=6, **kwargs):
		super().__init__()
		self.embeddings = nn.Embedding(20_000, emb_dims)
		self.CNN = nn.Sequential(
			FFTConv1d(emb_dims, hidden, 3),
			nn.Dropout(0.3),
			nn.ReLU(),
			FFTConv1d(hidden, hidden, 3),
			nn.Dropout(0.3),
			nn.ReLU(),
			nn.AdaptiveMaxPool1d(1),
		)
		self.linear = nn.Sequential(
			nn.Dropout(0.3),
			nn.Linear(hidden, hidden),
			nn.ReLU(),
			nn.Dropout(0.3),
			nn.Linear(hidden, classes),
		)

	def forward(self, x, lengths=None):
		if lengths is None:
			x, lengths = x
		x = self.embeddings(x)
		x = torch.transpose(x, 1, 2)
		x = self.CNN(x)
		return self.linear(x[...,0])
		
		
		
		
		
		
		
		
		
	