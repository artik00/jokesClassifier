import math
from text_utils import break_each_sentence_into_tokens, remove_repetition


class DataLoaderFromFile:

    def __init__(self, filename, max_number_of_sentences=None):

        self.filename = filename
        self._all_sentences = []
        self._all_sentences_splitted_by_word = []
        self._all_sentences_no_repetition = []
        self._max_number = math.inf if not max_number_of_sentences else max_number_of_sentences
        with open(filename, 'rb') as file:
            counter = 0
            for line in file:
                if counter < self._max_number:
                    self._all_sentences.append(line.decode(errors='ignore').strip())
                    counter += 1
        self._all_sentences_splitted_by_word = break_each_sentence_into_tokens(self._all_sentences)
        self._all_sentences_no_repetition = remove_repetition(self._all_sentences)

    def get_number_of_sentences(self):
        return self._max_number

    @staticmethod
    def get_all_sentences(self):
        return self._all_sentences

    def get_all_sentences_splitted_by_word(self):
        return self._all_sentences_splitted_by_word

    def get_all_sentences_without_repetition(self):
        return self._all_sentences_no_repetition
