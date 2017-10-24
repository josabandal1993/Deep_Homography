import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation, Input
from keras.layers import MaxPooling2D, Flatten, Dropout
from keras.models import Model
from keras import metrics
import numpy as np
import pickle
from keras import backend as K
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
import math

modelpath = 'saved_models/deephomography.11.h5'
filepath = "data_test/test_0.p" #File where you saved your test dataset

model = load_model(modelpath) 
file = open(filepath,"rb")
test = pickle.load(file)
file.close()

x_test = test['features']
y_test = test['labels'].astype('float32')
x_test = x_test.astype('float32') / 255
Mean_ave_corner_error = []

for i in range(test_samp):
	q = np.squeeze(model.predict_on_batch(x_test[i:i+1]))*32
	Corner_error_1 = math.sqrt(((q[0] - y_test[i][0])**2) + ((q[1] - y_test[i][1])**2))
	Corner_error_2 = math.sqrt(((q[2] - y_test[i][2])**2) + ((q[3] - y_test[i][3])**2))
	Corner_error_3 = math.sqrt(((q[4] - y_test[i][4])**2) + ((q[5] - y_test[i][5])**2))
	Corner_error_4 = math.sqrt(((q[6] - y_test[i][6])**2) + ((q[7] - y_test[i][7])**2))
	Corner_error = (Corner_error_1 + Corner_error_2 + Corner_error_3 + Corner_error_4)/4
	Mean_ave_corner_error.append(Corner_error)

Mean_ave_corner_error = np.average(Mean_ave_corner_error)
np.savetxt('losses.out',history.losses)
print("Mean Average Corner Error: ",Mean_ave_corner_error)
