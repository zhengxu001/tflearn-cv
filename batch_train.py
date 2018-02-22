import train
augmentations = ["NA", "FLIP"]
epochs = [60]
models = ["alex", "vgg", "res"]
for aug in augmentations:
	for epoch in epochs:
		for model in models:
			train.main(model+"-"+aug+"-"+str(epoch), epoch, aug, model)


self_models = ["alch11_without_dropout", "alch11", "alch19"]
for aug in augmentations:
	for epoch in epochs:
		for model in self_models:
			train.main(model+"-"+aug+"-"+str(epoch), epoch, aug, model)