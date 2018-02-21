import train
# augmentations = ["NA", "FLIP"]
augmentations = ["FLIP"]
# epochs = [20, 40, 60]
epochs = [50]
# models = ["alex", "vgg", "res", "alch"]
models = ["alex", "vgg", "res"]

for aug in augmentations:
	for epoch in epochs:
		for model in models:
			train.main(model+aug+epoch, epoch, aug, model)
			
