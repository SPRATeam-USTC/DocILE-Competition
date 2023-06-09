from libs.utils.vocab import DoclieonlykieEntityVocab
# dataset
train_npy_paths = ["/path/to/train_split/infos.npy"]
#no eval in training
valid_npy_paths = ["/path/to/val/infos.npy"]

Docliekie_vocab = DoclieonlykieEntityVocab()

# data augment
resize_type = 'fixed'

# classifier
classifier_layer_num = 1
