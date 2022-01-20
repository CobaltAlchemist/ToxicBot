from joblib import load
import numpy as np

model, cap = load('model_cap.joblib')
labels = ['toxic', 'severe_toxic', 'obscene', 'threat','insult','identity_hate']

def classify(x):
	x = [x]
	return cap.predict(np.concatenate((
        model.predict(x),
        model.steps[0][1].transform(x).toarray()), axis=1))[0]