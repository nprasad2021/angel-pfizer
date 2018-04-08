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
		'mizer':'SGD',
		'ensemble'=False}]

opt_temp = [{'dataset_pre':'melspectrograms',
		'batch_size':40,
		'num_classes':2,
		'num_epochs':10,
		'network':'audio_model',
		'input_shape':(224,224,3),
		'output_file':'outputs901.txt',
		'top_model':'top_init',
		'mizer':'SGD'}]

opt_ensemble = []
for data in datasets_pre:
	tmp = dict(opt_temp[0])

	tmp['ensemble'] = True
	tmp['dataset_pre'] = data
	tmp['network'] = ['vggnet', 'inceptionv3', 'resnet']
	tmp['freeze'] = 5
	tmp['num_epochs'] = 50
	tmp['output_file'] = 'output_ens.txt'

	opt_ensemble.append(tmp)

print(len(opt))
