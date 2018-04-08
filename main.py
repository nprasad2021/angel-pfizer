from helper import data_processing, networks
from keras.models import Model
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, Callback
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

opt = experiments.opt[ID]

dataset = opt['dataset_pre']
nnet = opt['network']
num_classes= opt['num_classes']
input_shape = (224,224,3)
batch_size=opt['batch_size']
epochs=opt['num_epochs']
num_frozen = opt['freeze']
output_path = opt['output_file']

archs = networks.all_nets()
print(nnet, dataset, num_frozen)

def run():

	base_model = archs[nnet](input_shape, num_frozen)
	top_model = networks.top_model(input_shape=base_model.output_shape[1:])
	model = Model(inputs=base_model.input, outputs=top_model(base_model.output))

	lr = data_processing.CustomLRScheduler(data_processing.lr_sched, verbose = 1)
	model.compile(optimizer=optimizers.SGD(), 
	                  loss='binary_crossentropy', 
	                  metrics=['accuracy'])

	training_generator, validation_generator = data_processing.get_gen(dataset)

	filepath = 'models/' + nnet + ".hdf5"
	tensorboard = TensorBoard(log_dir="logs/" + nnet + '/')
	es = EarlyStopping(min_delta=0.1, patience = 15)
	best_model_checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
	callbacks_list = [best_model_checkpoint, lr, es]

	nb_training_samples = 0
	nb_validation_samples = 0

	for ex in ['sick/', 'not_sick/']:
		nb_training_samples += len([name for name in os.listdir('data/' + dataset + '/train/' + ex) if os.path.isfile('data/' + dataset + '/train/' + ex + name)])
		nb_validation_samples += len([name for name in os.listdir('data/' + dataset + '/validation/' + ex) if os.path.isfile('data/' + dataset + '/validation/' + ex + name)])

	print(nnet, dataset)
	
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
	print(model.metrics_names)

	with open('outputs.txt', 'w') as f:
		f.write("accuracy:  " + str(acc[1]) + "   nnet: " + nnet + "  dataset: " + dataset + "  frozen: " + str(num_frozen))
	
	print(nnet, dataset)

if __name__ == "__main__":
	run()

