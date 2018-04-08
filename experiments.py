### Experiments

datasets_pre = ['melspectrograms', 'spectrograms', 'cwt_images']
archs = ['vggnet', 'resnet', 'inceptionv3','inception_res', 'vgg_by_hand']
top = ['top_init', 'top_long', 'top_short']
num_frozen = [0,3,5]


opt = [{'dataset_pre':'melspectrograms',
		'batch_size':40,
		'num_classes':2,
		'num_epochs':10,
		'network':'audio_model',
		'input_shape':(224,224,3),
		'freeze':0,
		'output_file':'outputs901.txt',
		'top_model':'top_init',
		'mizer':'SGD'}]

opt_temp = [{'dataset_pre':'melspectrograms',
		'batch_size':40,
		'num_classes':2,
		'num_epochs':10,
		'network':'audio_model',
		'input_shape':(224,224,3),
		'output_file':'outputs901.txt',
		'top_model':'top_init',
		'mizer':'SGD'}]

for fr in num_frozen:
	for ar in archs:
		for data in datasets_pre:
			for t in top:

				tmp = dict(opt_temp[0])

				tmp['dataset_pre'] = data
				tmp['network'] = ar
				tmp['num_epochs'] = 50
				tmp['freeze'] = fr
				tmp['output_file'] = 'outputs901.txt'
				tmp['top_model'] = t

				opt.append(tmp)

print(len(opt))