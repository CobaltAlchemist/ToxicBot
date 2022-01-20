import pickle
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, GlobalMaxPool1D
from tensorflow.keras.models import Model

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
labels = ['toxic', 'severe_toxic', 'obscene', 'threat','insult','identity_hate']

if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")

with open('tokenizer.pickle', 'rb') as f:
	tokenizer = pickle.load(f)

inputLayer = Input(shape=(None,))
embedLayer = Embedding(input_dim=20000, output_dim=128)(inputLayer)
lstmLayer = LSTM(units=60, return_sequences=True)(embedLayer)
maxPool = GlobalMaxPool1D()(lstmLayer)
dropOut1 = Dropout(0.1)(maxPool)
fcLayer1 = Dense(units=50, activation='relu')(dropOut1)
dropOut2 = Dropout(0.1)(fcLayer1)
fcLayer2 = Dense(units=6, activation='sigmoid')(dropOut2)
model = Model(inputs=inputLayer, outputs=fcLayer2)
model.load_weights("tensorflow_model.h5")

def classify(x):
	try:
		return np.array(model.predict(tokenizer.texts_to_sequences([x]))[0]) > 0.5
	except KeyError:
		return np.zeros((6,))