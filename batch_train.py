import train
# augmentations = ["NA", "FLIP"]
# augmentations = ["FLIP"]
# epochs = [65]
# models = ["alex", "vgg", "res"]
# models = ["vgg", "res"]
# models = ["res"]
# for aug in augmentations:
# 	for epoch in epochs:
# 		for model in models:
# 			train.main(model+"-"+aug+"-"+str(epoch), epoch, aug, model)

# self_models = ["alch11_without_dropout", "alch11", "alch19"]
# for aug in augmentations:
# 	for epoch in epochs:
# 		for model in self_models:
# 			train.main(model+"-"+aug+"-"+str(epoch), epoch, aug, model)

train.main("alch11-FLIP-65", 65, "FLIP", "alch11")
train.main("alch19-FLIP-72", 72, "FLIP", "alch19")