from helper import data_processing, networks


num_classes=2
input_shape = (224,224,3)

models = networks.all_nets()

for m in models:

	base_model = models[m](num_classes, input_shape)
	base_model.summary()
