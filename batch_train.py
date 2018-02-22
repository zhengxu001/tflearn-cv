import train
# augmentations = ["NA", "FLIP"]
augmentations = ["FLIP"]
epochs = [2]
# models = ["alex", "vgg", "res"]
models = ["res"]
for aug in augmentations:
	for epoch in epochs:
		for model in models:
			train.main(model+aug+str(epoch), epoch, aug, model)
			
