import pandas as pd
import numpy as np
import os

class ToxicDatabase:
	def __init__(self, filename : str):
		self.columns = ['server', 'author', 'toxic', 'severe_toxic', 'obscene', 'threat','insult','identity_hate']
		self.class_weights = np.array([ 10.43356872, 100.04451411,  18.88637709, 333.83054393, 20.25783928, 113.57366548])
		self.file = filename
		self._load_data()
		
	def _load_data(self):
		if os.path.exists(self.file) and os.path.isfile(self.file):
			self.data = pd.read_csv(self.file)
		else:
			self.data = pd.DataFrame(columns = self.columns)
			self.data.to_csv(self.file, index=False)
			
	def _append_data(self, data):
		assert len(data) == len(self.columns), f"{data} is invalid input"
		df = pd.DataFrame([data], columns = self.columns)
		self.data = pd.concat([self.data, df])
			
	def add(self, server, author, preds):
		packed = [server, author, *preds]
		self._append_data(packed)
		with open(self.file, 'a') as f:
			f.write(','.join(map(str, packed)))
			f.write('\n')
			
	def _get_stats(self, data):
		ret = {}
		ret['count'] = len(data)
		if ret['count'] > 0:
			ret['sum'] = data[self.columns[2:]].astype('int64').to_numpy().sum(axis=0)
			ret['avg'] = data[self.columns[2:]].astype('int64').to_numpy().mean(axis=0)
			ret['top_cat'] = self.columns[2:][ret['avg'].argmax()]
			ret['top_cat_weighted'] = self.columns[2:][(ret['avg']*self.class_weights).argmax()]
		return ret
		
	def _get_user_stats(self, data, author):
		return self._get_stats(data[data['author'] == author])
		
	def _get_server_stats(self, data):
		ret = self._get_stats(data)
		count = data[['author', 'server']].groupby(['author']).agg('count')
		count = list(zip(count.index.values, count['server'].tolist()))
		ret['most_toxic'] = list(sorted(count, reverse=True, key=lambda x: x[1]))
		return ret
			
	def user_stats(self, server, author):
		view = self.data[self.data['server'] == server]
		return self._get_user_stats(view, author)
		
	def server_stats(self, server):
		view = self.data[self.data['server'] == server]
		authors = view['author'].unique()
		ret = {}
		for author in authors:
			ret[author] = self._get_user_stats(view, author)
		ret[server] = self._get_server_stats(view)
		return ret
			
	def __len__(self):
		return len(self.data)
			
	def __repr__(self):
		return str(self.data)
		
if __name__ == "__main__":
	def random_entry():
		return np.random.uniform(0, 1, size=(6,)) > 0.5
	db = ToxicDatabase('unit_test.csv')
	all([db.add('dregs', 'corey', random_entry()) for _ in range(10)])
	all([db.add('dregs', 'kripa', random_entry()) for _ in range(20)])
	all([db.add('dregs', 'gizmo', random_entry()) for _ in range(15)])
	all([db.add('dndankmemes', 'crispy', random_entry()) for _ in range(15)])
	all([db.add('dndankmemes', 'cody', random_entry()) for _ in range(20)])
	all([db.add('dndankmemes', 'alex', random_entry()) for _ in range(10)])
	all([db.add('dndankmemes', 'corey', random_entry()) for _ in range(5)])
	print("---------user stats dregs 1---------")
	print(db.user_stats('dregs', 'corey'))
	print(db.user_stats('dregs', 'kripa'))
	print(db.user_stats('dregs', 'gizmo'))
	print("---------user stats dnd 1---------")
	print(db.user_stats('dndankmemes', 'crispy'))
	print(db.user_stats('dndankmemes', 'corey'))
	print("---------user stats who 1---------")
	print(db.user_stats('dregs', 'who'))
	print("---------server stats real 1---------")
	print(db.server_stats('dregs'))
	print(db.server_stats('dndankmemes'))
	print("---------server stats fake 1---------")
	print(db.server_stats('who'))
	db = ToxicDatabase('unit_test.csv')
	print("---------user stats dregs 2---------")
	print(db.user_stats('dregs', 'corey'))
	print(db.user_stats('dregs', 'kripa'))
	print(db.user_stats('dregs', 'gizmo'))
	print("---------user stats dnd 2---------")
	print(db.user_stats('dndankmemes', 'crispy'))
	print(db.user_stats('dndankmemes', 'corey'))
	print("---------user stats who 2---------")
	print(db.user_stats('dregs', 'who'))
	print("---------server stats real 2---------")
	print(db.server_stats('dregs'))
	print(db.server_stats('dndankmemes'))
	print("---------server stats fake 2---------")
	print(db.server_stats('who'))
	os.remove('unit_test.csv')