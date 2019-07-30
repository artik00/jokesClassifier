import pickle
import os
from ambiguity_vectorizer import AmbiguityVectorizer
from data_loader_from_file import DataLoaderFromFile
from interpersonal_vectorizer import InterpersonalVectorizer
from incogurity_vectorizer import IncogurityVectorizer
from phonetic_style import PhoneticStyle
import numpy as np
import gensim

word2vec_pretrained = 'word2vec/GoogleNews-vectors-negative300.bin.gz'
SHORT_RUN = True


if __name__ == "__main__":
    jokes_file = 'dataset/Jokes16000.txt'
    model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_pretrained,
                                                            binary=True, limit=50000)
    NUMBER_OF_SENTENCES = 10
    dataloader = DataLoaderFromFile(jokes_file, max_number_of_sentences=NUMBER_OF_SENTENCES)
    phonetic_vector = PhoneticStyle(dataloader.get_all_sentences()).vector
    jokes_splitted_by_word = dataloader.get_all_sentences_splitted_by_word()

    ambiguity_vectorizer = AmbiguityVectorizer(jokes_splitted_by_word)
    ambiguity_vector = ambiguity_vectorizer.get_features_vector()
    incogurity_vector = IncogurityVectorizer(dataloader.get_all_sentences_without_repetition(), model.similarity).vector
    interpersonal_vectorizer = InterpersonalVectorizer()
    interpersonal_vector = interpersonal_vectorizer.get_feature_vector(sentences=jokes_splitted_by_word)
    full_vector = list()
    temp = np.concatenate((phonetic_vector, ambiguity_vector, incogurity_vector, interpersonal_vector), axis=1)
    pass

