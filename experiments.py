### Experiments

datasets_pre = ['melspectrograms', 'spectrograms', 'cwt_images']
archs = ['simple_cnn', 'vggnet', 'resnet', 'inceptionv3',
         'inception_res', 'audio_model']

opt = [{'dataset_pre':'melspectrograms',
		'batch_size':40,
		'num_classes':2,
		'num_epochs':10,
		'network':'audio_model',
		'input_shape':(224,224,3)}]

opt_temp = [{'dataset_pre':'melspectrograms',
		'batch_size':40,
		'num_classes':2,
		'num_epochs':10,
		'network':'audio_model',
		'input_shape':(224,224,3)}]

for ar in archs:
	for data in datasets_pre:
		tmp = dict(opt_temp[0])
		tmp['dataset_pre'] = data
		tmp['network'] = ar
		opt.append(tmp)

print(len(opt))