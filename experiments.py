### Experiments


datasets_pre = ['melspectrograms', 'spectrograms', 'cwt_images']
archs = ['vggnet', 'resnet', 'inceptionv3',
         'inception_res', 'audio_model', 'audio_model_2', 'tim_model', 'simple_cnn', 'jaron']
num_frozen = [5,10,15]


opt = [{'dataset_pre':'melspectrograms',
		'batch_size':40,
		'num_classes':2,
		'num_epochs':10,
		'network':'audio_model',
		'input_shape':(224,224,3),
		'freeze':0,
		'output_file':'outputs144.txt'}]

opt_temp = [{'dataset_pre':'melspectrograms',
		'batch_size':40,
		'num_classes':2,
		'num_epochs':10,
		'network':'audio_model',
		'input_shape':(224,224,3),
		'output_file':'outputs144.txt'}]

for fr in num_frozen:
	for ar in archs:
		for data in datasets_pre:

			if fr > 5:
				if archs.index(ar) > 3:
					continue
			tmp = dict(opt_temp[0])
			tmp['dataset_pre'] = data
			tmp['network'] = ar
			tmp['num_epochs'] = 50
			tmp['freeze'] = fr
			tmp['output_file'] = 'outputs144.txt'
			opt.append(tmp)

print(len(opt))