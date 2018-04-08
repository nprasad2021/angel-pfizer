# Deep Learning to Classify Respiratory Sounds

2nd place, Digital Health Challenge at the CMG Hackathon

Built a deep learning classifier to differentiate between sick and not-sick patients.
Final model - 87.6% accuracy on test set

* Melspectrogram data
* Image Transformations (normalize mean, variance)
* Base: VGG Model pretrained w/ ImageNet
* Top: Densely connected network w/ Dropout
* Optimizer: SGD w/ learning rate decay
* 5 Frozen layers

## TODO:
* Ensemble Learning: Combine multiple models prior to densely connected block
* Model Averages: Take models trained on different models, average final results to estimate condition




