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
# train.main("alex-FLIP-60", 60, "FLIP", "alex")
# train.main("vgg-NA-65-shuffle", 65, "NA", "vgg")
train.main("vgg-FLIP-65-shuffle", 65, "FLIP", "vgg")
train.main("vgg13", 65, "FLIP", "vgg13")
train.main("vgg16", 65, "FLIP", "vgg16")

# self_models = ["alch11_without_dropout", "alch11", "alch19"]
# for aug in augmentations:
# 	for epoch in epochs:
# 		for model in self_models:
# 			train.main(model+"-"+aug+"-"+str(epoch), epoch, aug, model)