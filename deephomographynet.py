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
import time
import os
import math

batch_size = 64
dropout = 0.5
epochs = 12 
steps_per_epoch = 7900
test_samp = 5000

time1 = time.time()

def val_generator(batch_size):
	while True:
		n = 0
		filename = "data_val/val_"+str(n)+".p"
		file = open(filename,"rb")
		val = pickle.load(file)
		file.close()
		x_val = val['features']
		y_val = val['labels']
		x_val = x_val.astype('float32') / 255
		y_val = y_val.astype('float32')	/ 32
		for n in range(100):
			yield x_val[n*batch_size:(n+1)*batch_size], y_val[n*batch_size:(n+1)*batch_size]

def train_generator(batch_size):
	m = 0
	while True:
		n = 79
		for i in range(n):
			filename = "data_train/train_"+str(i)+".p"
			file = open(filename,"rb")
			train = pickle.load(file)
			file.close()
			x_train = train['features']
			y_train = train['labels']
			x_train = x_train.astype('float32') / 255
			y_train = y_train.astype('float32') / 32
			for j in range(100):
				yield x_train[j*batch_size:(j+1)*batch_size], y_train[j*batch_size:(j+1)*batch_size]
			print("  ",(m+1)*100," Iterations Completed")
			m=m+1

def lr_schedule(epoch):
	lr = 5e-3
	if epoch > 7:
		lr = 5e-5
	elif epoch > 3:
		lr = 5e-4
	print("Learning rate: ", lr)
	return lr
 	
def euclidean_loss(y_true, y_pred):
	global batch_size
	return 0.5*(K.sum(K.square(y_pred - y_true), axis=-1, keepdims=True))

def euclidean_distance(y_true, y_pred):
	global batch_size
	return K.sqrt(K.maximum(K.sum(K.square(y_true - y_pred), axis=-1, keepdims=True), K.epsilon()))

class LossHistory(keras.callbacks.Callback):
	def on_train_begin(self, logs={}):
		self.losses = []
	def on_batch_end(self, batch, logs={}):
		self.losses.append(logs.get('loss'))


###############Start of the Model######################

input_shape = (128, 128, 2)

inputs = Input(shape = input_shape)

y = Conv2D(filters = 64, kernel_size = 3, strides = (1,1), padding = 'same', name = 'conv1')(inputs)
y = BatchNormalization()(y)
y = Activation('relu')(y)

y = Conv2D(filters = 64, kernel_size = 3, strides = (1,1), padding = 'same', name = 'conv2')(y)
y = BatchNormalization()(y)
y = Activation('relu')(y)

y = MaxPooling2D(pool_size = (2,2), strides = (2,2), name = 'pool1')(y)

y = Conv2D(filters = 64, kernel_size = 3, strides = (1,1), padding = 'same', name = 'conv3')(y)
y = BatchNormalization()(y)
y = Activation('relu')(y)

y = Conv2D(filters = 64, kernel_size = 3, strides = (1,1), padding = 'same', name = 'conv4')(y)
y = BatchNormalization()(y)
y = Activation('relu')(y)

y = MaxPooling2D(pool_size = (2,2), strides = (2,2), name = 'pool2')(y)

y = Conv2D(filters = 128, kernel_size = 3, strides = (1,1), padding = 'same', name = 'conv5')(y)
y = BatchNormalization()(y)
y = Activation('relu')(y)

y = Conv2D(filters = 128, kernel_size = 3, strides = (1,1), padding = 'same', name = 'conv6')(y)
y = BatchNormalization()(y)
y = Activation('relu')(y)

y = MaxPooling2D(pool_size = (2,2), strides = (2,2), name = 'pool3')(y)

y = Conv2D(filters = 128, kernel_size = 3, strides = (1,1), padding = 'same', name = 'conv7')(y)
y = BatchNormalization()(y)
y = Activation('relu')(y)

y = Conv2D(filters = 128, kernel_size = 3, strides = (1,1), padding = 'same', name = 'conv48')(y)
y = BatchNormalization()(y)
y = Activation('relu')(y)

y = Flatten()(y)
y = Dropout(dropout, name = 'dropout1')(y)

y = Dense(1024, name = 'fc1', activation = 'relu')(y)
y = Dropout(dropout, name = 'dropout2')(y)
y_out = Dense(8, name = 'fc2', activation= 'relu')(y)

model = Model(inputs = inputs, outputs = y_out)

model.summary()

model.compile(optimizer = SGD(momentum = 0.9, lr = 0.005, decay = 0),
			loss = 'mse')

save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'deephomography.{epoch:02d}.h5'
filepath = os.path.join(save_dir, model_name)

checkpoint = ModelCheckpoint(filepath=filepath,verbose = 1)

lr_scheduler = LearningRateScheduler(lr_schedule)

history = LossHistory()

callbacks = [checkpoint, lr_scheduler, history]

model.fit_generator(train_generator(batch_size), epochs=epochs,steps_per_epoch=steps_per_epoch,
					validation_data = val_generator(batch_size), validation_steps = 100,
					callbacks = callbacks, shuffle=False)

print("/nTotal training time = %.3f"%((time.time()-time1)/60),"mins")
