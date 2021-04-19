import sys

sys.path.append('/users/PCCH0011/cch0017/PROJECTS/Rafi_dynamic_site_prediction_2020')
from ops import *
import keras as K

K.backend.set_image_data_format('channels_last')

model_name = 'BPnet-like'


def create_model(seq_length=500, dropout_rate=0.01, output_dims=3):
	_inputs = K.layers.Input(shape=(seq_length, 4))

	# tfbs scan
	_next_repr = K.layers.Conv1D(filters=64,
								 kernel_size=25,
								 padding='same',
								 activation='relu')(_inputs)
	# residual dilated convolution
	for i in range(1, 10):
		tmp = K.layers.Conv1D(filters=64,
							  kernel_size=3,
							  padding='same',
							  activation='relu',
							  dilation_rate=2 ** i)(_next_repr)
		_next_repr = K.layers.add([tmp, _next_repr])
		_next_repr = K.layers.Dropout(rate=dropout_rate)(_next_repr)
	# compression
	for i in range(7):
		_next_repr = K.layers.Conv1D(filters=128,
									 kernel_size=2,
									 strides=2,
									 padding='valid',
									 activation='relu')(_next_repr)
		_next_repr = K.layers.Dropout(rate=dropout_rate)(_next_repr)
	# output
	_next_repr = K.layers.Flatten()(_next_repr)
	_next_repr = K.layers.Dense(units=output_dims,
								activation='softmax')(_next_repr)
	# model
	model = K.Model(inputs=_inputs,
					outputs=_next_repr)

	return model
