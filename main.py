from helper import data_processing, networks
from keras.models import Model
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import optimizers
import os
import os.path
import sys

ROOT_PATH = sys.argv[1]
print(ROOT_PATH)
os.chdir(ROOT_PATH)

num_classes=2
input_shape = (224,224,3)
batch_size=60
epochs=10
dataset = 'melspectrograms'

archs = networks.all_nets()

for m in archs:
	print(m)
	base_model = archs[m](input_shape)
	top_model = networks.top_model(input_shape=base_model.output_shape[1:])
	model = Model(inputs=base_model.input, outputs=top_model(base_model.output))
	model.compile(optimizer=optimizers.Adam(), 
                      loss='binary_crossentropy', 
                      metrics=['accuracy'])
	print(os.getcwd())
	training_generator, validation_generator = data_processing.get_gen(dataset)

	filepath = 'models/' + m
	tensorboard = TensorBoard(log_dir="logs/" + m)

	best_model_checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
	callbacks_list = [best_model_checkpoint]

	nb_training_samples = 0
	nb_validation_samples = 0

	for ex in ['sick/', 'not_sick/']:
		nb_training_samples += len([name for name in os.listdir('data/' + dataset + '/train/' + ex) if os.path.isfile('data/' + dataset + '/train/' + ex + name)])
		nb_validation_samples += len([name for name in os.listdir('data/' + dataset + '/validation/' + ex) if os.path.isfile('data/' + dataset + '/validation/' + ex + name)])

	print(nb_training_samples, nb_validation_samples)

	model.fit_generator(
    	training_generator,
    	steps_per_epoch=nb_training_samples/batch_size,
    	epochs=epochs,
    	validation_data=validation_generator,
    	validation_steps=nb_validation_samples/batch_size,
    	callbacks=callbacks_list,
    	verbose=2)















