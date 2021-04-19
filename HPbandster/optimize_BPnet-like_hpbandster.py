import sys
import warnings
warnings.filterwarnings('ignore')
sys.path.append('/users/PCCH0011/cch0017/PROJECTS/Rafi_dynamic_site_prediction_2020')
from bedclass import *
from network_BPnet_like_hp_optim_hpbandster import *
from motif import *
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedKFold
from keras.utils import to_categorical
from ops import *

from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import classification_report
import subprocess
from keras.callbacks import LearningRateScheduler

# hpbandster
import pickle
import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres
from hpbandster.optimizers import HyperBand
from hpbandster.core.worker import Worker
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
#
import logging

import argparse

# settings
datasets = ['notch1','notch2','rbpj']

build = 'mm9'
sequence_length = 1000
use_bgs = [False]
bg_map = {True: 'with_bg',
		  False: 'no_bg'}

#
model_name = model_name

parser = argparse.ArgumentParser(description='.')
parser.add_argument('--dataset', type=str, action='store', dest='dataset', help='dataset to use')
parser.add_argument('--shared_dir', type=str, action='store', dest='shared_dir', help='output folder')
parser.add_argument('--sample_size', type=int, action='store', dest='sample_size', default=15, help='# of samples to draw')
parser.add_argument('--max_budget', type=int, action='store', dest='max_budget', default=50, help='maximum iteration to train each single model')
parser.add_argument('--min_budget', type=int, action='store', dest='min_budget', default=10, help='minimum iteration to train each single model')
parser.add_argument('--run_id', type=str, action='store', dest='run_id', default='hpbandster_run_1', help='output folder')

args = parser.parse_args()

# keras worker
class KerasWorker(Worker):
	def __init__(self, dataset=args.dataset,use_bg=False,sequence_length=1000,**kwargs):
		super().__init__(**kwargs)

		self.dataset=dataset
		self.use_bg=use_bg
		self.sequence_length=sequence_length

		# load toy set
		test_df = pd.read_csv('test_tracks.txt', header=0, sep='\t')
		mouse_feature_set_df = pd.read_csv(os.path.join(data_path, 'Feature_tracks/mm9/tracks.txt'),
										   header=None, sep='\t', index_col=None)
		mouse_feature_set_df.rename(axis=1, mapper={0: 'track', 1: 'cell', 2: 'signal', 3: 'index'},
									inplace=True)
		test_df = pd.merge(test_df, mouse_feature_set_df, on=['track', 'cell', 'signal'], how='left')
		#test_df['index'].astype(int)
		print(test_df)
		grouped_test_df = test_df.groupby('group')

		if args.dataset in datasets:
			print('starting training: ', self.dataset, 'using bg: ', self.use_bg)
			self.x_train, self.y_train, self.x_valid, self.y_valid = track.get_X_n_y(data_set=self.dataset,
											   sequence_length=self.sequence_length,
											   valid=True,test_ratio=0.2,
											   use_bg=self.use_bg,
											   random_state=8
											   )
		else:
			group=int(args.dataset.split('_')[1])
			group_data = grouped_test_df.get_group(group)
			print('starting training: ', 'group_{}'.format(group), 'using bg: ', self.use_bg)
			self.x_train, self.y_train, self.x_valid, self.y_valid = track.get_X_n_y(data_set=args.dataset,
											 sequence_length=self.sequence_length,
											 random_state=8,
											 valid=True, test_ratio=0.2,
											 use_bg=self.use_bg,
											 pos_file=os.path.join(data_path, 'Feature_tracks/mm9/data/{}'.format(group_data[group_data['class'] == 'pos']['index'].values[0])),
											 neg_file=os.path.join(data_path, 'Feature_tracks/mm9/data/{}'.format(group_data[group_data['class'] == 'neg']['index'].values[0])),
											 assign_class_label=True
											 )
		# y_train is with 1/0 for positive/negative and -1 for nearby negative labels
		if self.use_bg:
			self.y_train = self.y_train + 1  # this sets all label to start from 0, [0, 1, 2] with 2 being positive, 1 being negative, and 0 being nearby negative
			self.y_valid = self.y_valid + 1
		# otherwise this sets to 0 and 1, for 1 being positive and 0 being negative

		# class weights
		self.class_weights = dict(zip(np.unique(self.y_train), compute_class_weight('balanced', np.unique(self.y_train), self.y_train)))
		self.output_dims = len(self.class_weights)
		print('output dimensions: ', self.output_dims)
		# one hot target
		self.y_train = to_categorical(self.y_train, num_classes=self.output_dims)
		self.y_valid = to_categorical(self.y_valid, num_classes=self.output_dims)

	def compute(self, config, budget, **kwargs):
		"""
		Simple example for a compute function using a feed forward network.
		It is trained on the MNIST dataset.
		The input parameter "config" (dictionary) contains the sampled configurations passed by the bohb optimizer
		"""

		K.backend.clear_session()
		model= create_model(seq_length=self.sequence_length,
									output_dims=self.output_dims,
									hp_config=config)

		def step_decay(epoch, hp_config=config):
			initial_lrate = hp_config['learning_rate']
			drop = 0.5
			epochs_drop = 10.0
			lrate = initial_lrate * np.math.pow(drop, np.math.floor((1 + epoch) / epochs_drop))
			return lrate

		lrate = LearningRateScheduler(step_decay,)
		#
		callbacks = [lrate,
					 K.callbacks.EarlyStopping(monitor='val_categorical_crossentropy',
											   patience=10,
											   restore_best_weights=True),
					 ]
		# model parameters
		history=model.fit(x=self.x_train, y=self.y_train,
				  batch_size=32,
				  epochs=int(budget),
				  verbose=0,
				  validation_data=(self.x_valid, self.y_valid),
				  class_weight=self.class_weights,
				  callbacks=callbacks
				  )
		# maybe need to manually set best model parameter
		val_loss,val_categorical_crossentropy,val_acc=model.evaluate(self.x_valid, self.y_valid,verbose=0)
		'''categorical_crossentropy','accuracy'''
		# use keras training history to return the loss
		return ({
			'loss': 1- val_acc,  # remember: HpBandSter always minimizes!
			'info': {
				'loss_hist': history.history['loss'],
				'accuracy_hist':history.history['acc'],

				'val_loss_hist': history.history['val_categorical_crossentropy'],
				'val_acc_hist': history.history['val_acc'],

				'val_loss': val_loss,
				'val_acc': val_acc,

				'epochs':len(history.history['acc']),
				'early_stop': model.stop_training
				}
			}
		)

	# hp config settings
	@staticmethod
	def get_configspace():
		"""
		It builds the configuration space with the needed hyperparameters.
		It is easily possible to implement different types of hyperparameters.
		Beside float-hyperparameters on a log scale, it is also able to handle categorical input parameter.
		:return: ConfigurationsSpace-Object


		hp_config = {

		'dropout_rate': tune.quniform(0.01, 0.2, 0.02),  # truncated normal from 0 to 0.2, with 0.01 increment

		'tfbs_kernel_size': tune.choice([32, 64, 128, 256, 512]),
		'tfbs_kernel_length': tune.choice([10, 25, 40]),

		'dilation_level': tune.choice([6, 7, 8, 9, 10, ]),
		'dilation_kernel_length': tune.randint(2, 6),

		'compression_level': tune.choice([4, 5, 6, 7]),
		'compression_kernel_size': tune.choice([64, 128, 256]),
		'learning_rate': tune.qloguniform(0.0001, 0.1, 0.0001),
		}
		"""
		cs = CS.ConfigurationSpace()

		dropout_rate = CSH.UniformFloatHyperparameter('dropout_rate', lower=0.01, upper=0.1, default_value=0.05, log=False, q=0.01)

		tfbs_filter_size= CSH.OrdinalHyperparameter('tfbs_filter_size',sequence=[32,64,128])
		tfbs_kernel_length= CSH.OrdinalHyperparameter('tfbs_kernel_length', sequence=[10,25])

		dilation_level= CSH.OrdinalHyperparameter('dilation_level', sequence=[4,6,8,10])
		dilation_kernel_length = CSH.UniformIntegerHyperparameter('dilation_kernel_length',lower=2,upper=6,default_value=2,log=False)

		compression_filter_size = CSH.OrdinalHyperparameter('compression_filter_size', sequence=[64, 128, 256])

		learning_rate = CSH.UniformFloatHyperparameter('learning_rate', lower=0.0001, upper=0.01, default_value=0.001, log=True)

		#
		cs.add_hyperparameters([dropout_rate,
								tfbs_filter_size,
								tfbs_kernel_length,
								dilation_level,
								dilation_kernel_length,
								compression_filter_size,
								learning_rate])

		return cs

#
shared_dir=args.shared_dir

# results logger
result_logger = hpres.json_result_logger(directory=shared_dir,
										 overwrite=False)
# initialize name server
run_id=args.run_id
NS = hpns.NameServer(run_id=run_id,
					 host='127.0.0.1',
					 working_directory=shared_dir,
					 )
ns_host, ns_port = NS.start()
# define worker
worker = KerasWorker(dataset=args.dataset,
					 use_bg=False,
					 sequence_length=1000,
					 host='127.0.0.1',
					 run_id=run_id,
					 nameserver=ns_host,
					 nameserver_port=ns_port,
					 timeout=120
					 )
worker.run(background=True)

# define hp optimizer
optimizer = HyperBand(configspace=KerasWorker.get_configspace(),
					  run_id=run_id,
					  host='127.0.0.1',
					  nameserver=ns_host,
					  nameserver_port=ns_port,
					  result_logger=result_logger,
					  min_budget=args.min_budget,
					  max_budget=args.max_budget
					  )
# run
result = optimizer.run(n_iterations=args.sample_size)

# save results
with open(os.path.join(shared_dir,'results.pkl'), 'wb') as fh:
	pickle.dump(result, fh)

# close optimizer & nameserver
optimizer.shutdown()
NS.shutdown()
