import os

class DataLoaderFromFile:

    def __init__(self, filename):
        self.filename = filename
        self.all_sentences = []
        self.all_sentences_splitted_by_word = []
        with open(filename, 'rb') as file:
            for line in file:
                self.all_sentences.append(line.decode(errors='ignore').strip())
        self.break_each_sentence_into_tokens()

    def get_all_sentences(self):
        return self.all_sentences

    def get_all_sentences_splitted_by_word(self):
        return self.all_sentences_splitted_by_word

    def break_each_sentence_into_tokens(self):
        for sentence in self.all_sentences:
            words = sentence.split()
            new_sentence = []
            for word in words:
                new_sentence.append(word)
            self.all_sentences_splitted_by_word.append(new_sentence)
