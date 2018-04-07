from helper import data_processing, networks
from keras.models import Model


num_classes=2
input_shape = (224,224,3)

archs = networks.all_nets()

for m in archs:
	print(m)
	base_model = archs[m](num_classes, input_shape)
	print(base_model.output_shape)
	top_model = networks.top_model(input_shape=base_model.output_shape[1:])
	model = Model(inputs=base_model.input, outputs=top_model(base_model.output))
	model.summary()










