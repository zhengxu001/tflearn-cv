import train
augmentations = ["NA", "FLIP"]
epochs = [60]
models = ["alex", "vgg", "res"]
for aug in augmentations:
	for epoch in epochs:
		for model in models:
			train.main(model+"-"+aug+"-"+str(epoch), epoch, aug, model)
			
