import numpy as np

class DocTypeVocab:
    key_words = []

    def __init__(self):
        self._words_ids_map = dict()
        self._ids_words_map = dict()

        for word_id, word in enumerate(self.key_words):
            self._words_ids_map[word] = word_id
            self._ids_words_map[word_id] = word
    
    def __len__(self):
        return len(self._words_ids_map)

    def word_to_id(self, word):
        return self._words_ids_map[word]

    def words_to_ids(self, words):
        return [self.word_to_id(word) for word in words]

    def id_to_word(self, word_id):
        return self._ids_words_map[word_id]
    
    def ids_to_words(self, words_id):
        return [self.id_to_word(word_id) for word_id in words_id]

easyocr_vocab = np.load('../resources/easyocr_vocab.npy').tolist()

class EasyOcrVocab(DocTypeVocab):
    key_words = easyocr_vocab