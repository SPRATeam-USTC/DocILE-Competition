from libs.utils.vocab import DocTypeVocab
from libs.utils.counter import Counter



# dataset path dict
datasets = dict()

# document type vocab
doctype_vocab = DocTypeVocab()

# counter for show each item loss
counter = Counter(cache_nums=1000)