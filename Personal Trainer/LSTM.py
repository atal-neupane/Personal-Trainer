from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.model_selection import train_test_split
import numpy as np 
import matplotlib.pyplot as plt
import EstimatePoseModule as epm

detector = epm.poseDetector()

model = Sequential()
model.add(LSTM((1), batch_input_shape=(None, 3, 1), return_sequences=False))
model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])
model.summary()

history = model.fit()