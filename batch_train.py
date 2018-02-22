import train
# augmentations = ["NA", "FLIP"]
augmentations = ["FLIP"]
# epochs = [20, 40, 60]
# epochs = [50]
epochs = [2]
# models = ["alex", "vgg", "res", "alch"]
models = ["alex", "vgg", "res"]
# models = ["vgg", "res"]
for aug in augmentations:
	for epoch in epochs:
		for model in models:
			train.main(model+aug+str(epoch), epoch, aug, model)
			
