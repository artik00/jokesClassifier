import pickle
import os

import gensim

from data_loader_from_git import DataLoader
from ambiguity_vectorizer import AmbiguityVectorizer
from data_loader_from_file import DataLoaderFromFile
from incogurity_vectorizer import IncogurityVectorizer
from phonetic_style import PhoneticStyle

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
    # jokes_splitted_by_word = dataloader.get_all_sentences_splitted_by_word()
    # ambiguityVectorizerFeatures = AmbiguityVectorizer(jokes_splitted_by_word)
    # incogurityVector = IncogurityVectorizer(dataloader.all_sentences_no_repetition, model.similarity).vector
    PhoneticStyle = PhoneticStyle(dataloader.all_sentences).vector
    pass

