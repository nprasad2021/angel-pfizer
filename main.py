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

class CustomLRScheduler(Callback):

    def __init__(self, schedule, verbose = True):
        super(CustomLRScheduler, self).__init__()
        self.schedule = schedule
        self.verbose = verbose

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        last_lr = K.get_value(self.model.optimizer.lr)
        lr = self.schedule(last_lr)
        if self.verbose:
            print(f"New learning rate is {lr}")
        if not isinstance(lr, (float, np.float32, np.float64)):
            raise ValueError('The output of the "schedule" function '
                             'should be float.')
        K.set_value(self.model.optimizer.lr, lr)

def lr_sched(last_lr):

    return 0.99*last_lr

archs = networks.all_nets()
def run():

	base_model = archs[nnet](input_shape)
	top_model = networks.top_model(input_shape=base_model.output_shape[1:])
	model = Model(inputs=base_model.input, outputs=top_model(base_model.output))
	lr = CustomLRScheduler(lr_sched, verbose = 1)
	model.compile(optimizer=optimizers.SGD(), 
	                  loss='binary_crossentropy', 
	                  metrics=['accuracy'])

	training_generator, validation_generator = data_processing.get_gen(dataset)

	filepath = 'models/' + nnet + ".hdf5"
	tensorboard = TensorBoard(log_dir="logs/" + nnet + '/')

	best_model_checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
	callbacks_list = [best_model_checkpoint, lr]

	nb_training_samples = 0
	nb_validation_samples = 0

	for ex in ['sick/', 'not_sick/']:
		nb_training_samples += len([name for name in os.listdir('data/' + dataset + '/train/' + ex) if os.path.isfile('data/' + dataset + '/train/' + ex + name)])
		nb_validation_samples += len([name for name in os.listdir('data/' + dataset + '/validation/' + ex) if os.path.isfile('data/' + dataset + '/validation/' + ex + name)])

	print(nnet)
	model.fit_generator(
		training_generator,
		steps_per_epoch=nb_training_samples/batch_size,
		epochs=epochs,
		validation_data=validation_generator,
		validation_steps=nb_validation_samples/batch_size,
		callbacks = callbacks_list,
		verbose=2)

if __name__ == "__main__":
	run()
















