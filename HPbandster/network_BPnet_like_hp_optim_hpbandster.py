import sys

sys.path.append('/users/PCCH0011/cch0017/PROJECTS/Rafi_dynamic_site_prediction_2020')
from ops import *
import keras as K
import numpy as np

K.backend.set_image_data_format('channels_last')
from keras.callbacks import LearningRateScheduler

model_name = 'BPnet-like-hp-optim-hpbandster'


def create_model(seq_length=500, output_dims=3, hp_config=None):

	dropout_rate = hp_config['dropout_rate']

	_inputs = K.layers.Input(shape=(seq_length, 4))

	# tfbs scan
	_next_repr = K.layers.Conv1D(filters=hp_config['tfbs_filter_size'],
								 kernel_size=hp_config['tfbs_kernel_length'],
								 padding='same',
								 activation='relu'
								 )(_inputs)
	# residual dilated convolution
	for i in range(1, hp_config['dilation_level']):
		tmp = K.layers.Conv1D(filters=hp_config['tfbs_filter_size'],
							  kernel_size=hp_config['dilation_kernel_length'],
							  padding='same',
							  activation='relu',
							  dilation_rate=2 ** i
							  )(_next_repr)
		_next_repr = K.layers.add([tmp, _next_repr])
		_next_repr = K.layers.Dropout(rate=dropout_rate)(_next_repr)
	# compression
	for i in range(7):
		_next_repr = K.layers.Conv1D(filters=hp_config['compression_filter_size'],
									 kernel_size=2,
									 strides=2,
									 padding='valid',
									 activation='relu')(_next_repr)
		_next_repr = K.layers.Dropout(rate=dropout_rate)(_next_repr)
	# output
	_next_repr = K.layers.Flatten()(_next_repr)
	_next_repr = K.layers.Dense(units=output_dims, activation='softmax')(_next_repr)
	# model
	model = K.Model(inputs=_inputs, outputs=_next_repr)

	model.compile(optimizer=K.optimizers.adam(lr=hp_config['learning_rate']),
				  loss='categorical_crossentropy',
				  metrics=['categorical_crossentropy','accuracy',]
				 )

	return model
