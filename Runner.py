import pickle
import os

import gensim

from DataLoader import DataLoader
from AmbiguityVectorizer import AmbiguityVectorizer
from DataLoaderFromFile import DataLoaderFromFile
from IncogurityVectorizer import IncogurityVectorizer

word2vec_pretrained = '/Users/amirl/Downloads/GoogleNews-vectors-negative300.bin'
SHORT_RUN = True
if __name__ == "__main__":
    model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_pretrained,
                                                            binary=True, limit=50000)
    jokes_file = 'dataset/Jokes16000.txt'
    dataloader = DataLoaderFromFile(jokes_file)
    if SHORT_RUN:
        dataloader.all_sentences = dataloader.all_sentences[:100]
        dataloader.all_sentences_splitted_by_word = dataloader.all_sentences_splitted_by_word[:100]
    jokes_splitted_by_word = dataloader.get_all_sentences_splitted_by_word()
    ambiguityVectorizer = AmbiguityVectorizer(jokes_splitted_by_word)
    incogurityVector = IncogurityVectorizer(jokes_splitted_by_word, model.similarity).vector
    pass

