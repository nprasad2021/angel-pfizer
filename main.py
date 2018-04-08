from helper import data_processing, networks
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, Callback
from keras.layers import Average
from keras import optimizers
import os
import os.path
import sys
import experiments
from keras import backend as K
import numpy as np

ROOT_PATH = str(sys.argv[1])
ID = int(sys.argv[2])

print(ID)
os.chdir(ROOT_PATH)

opt = experiments.opt_ensemble[ID]

nnet = opt['network']
top_model = opt['top_model']
dataset = opt['dataset_pre']
num_classes= opt['num_classes']
input_shape = (224,224,3)	
output_path = opt['output_file']
num_frozen = opt['freeze']
batch_size=opt['batch_size']
epochs=opt['num_epochs']

all_top = networks.all_top()
archs = networks.all_nets()
print(nnet, dataset, num_frozen, top_model)


def run():
	if not opt['ensemble']:
		base_model = archs[nnet](input_shape, num_frozen)
		top_modality = all_top[top_model](input_shape=base_model.output_shape[1:])
		model = Model(inputs=base_model.input, outputs=top_modality(base_model.output))
	else:

		model = networks.ensemble(nnet, input_shape, num_frozen)



	lr = data_processing.CustomLRScheduler(data_processing.lr_sched, verbose = 1)
	model.compile(optimizer=optimizers.SGD(), 
	                  loss='binary_crossentropy', 
	                  metrics=['accuracy'])

	training_generator, validation_generator = data_processing.get_gen(dataset)
	if not opt['ensemble']:
		filepath = 'models/' + str(num_frozen) + '/' + top_model + '/' + dataset + '/'
		best_model_checkpoint = ModelCheckpoint(filepath + nnet + ".hdf5", monitor='val_acc', verbose=1, save_best_only=True, mode='max')
	else:
		filepath = 'models/ensemble/'
		best_model_checkpoint = ModelCheckpoint(filepath + dataset + ".hdf5", monitor='val_acc', verbose=1, save_best_only=True, mode='max')
	
	if not os.path.exists(filepath):
		os.makedirs(filepath)

	tensorboard = TensorBoard(log_dir="logs/" + nnet + '/')
	es = EarlyStopping(min_delta=0.1, patience = 15)

	
	callbacks_list = [best_model_checkpoint, lr, es]

	nb_training_samples = 0
	nb_validation_samples = 0

	for ex in ['sick/', 'not_sick/']:
		nb_training_samples += len([name for name in os.listdir('data/' + dataset + '/train/' + ex) if os.path.isfile('data/' + dataset + '/train/' + ex + name)])
		nb_validation_samples += len([name for name in os.listdir('data/' + dataset + '/validation/' + ex) if os.path.isfile('data/' + dataset + '/validation/' + ex + name)])

	print(dataset)
	
	model.fit_generator(
		training_generator,
		steps_per_epoch=nb_training_samples/batch_size,
		epochs=epochs,
		validation_data=validation_generator,
		validation_steps=nb_validation_samples/batch_size,
		callbacks = callbacks_list,
		verbose=2)
	
	acc = model.evaluate_generator(
		validation_generator,
		steps=nb_validation_samples/batch_size)

	print(acc)

	with open(output_path, 'a+') as f:
		if not opt['ensemble']:
			f.write("accuracy:  " + str(acc[1])[0:4] + "   nnet: " + nnet + "  dataset: " + dataset + "  frozen: " + str(num_frozen) + "   top: " + top_model)
			f.write('\n')
		else:
			f.write("accuracy:  " + str(acc[1])[0:4] + "  dataset: " + dataset + "  frozen: " + str(num_frozen))
			f.write('\n')
	
	print(nnet, dataset, num_frozen, top_model)

if __name__ == "__main__":
	run()

