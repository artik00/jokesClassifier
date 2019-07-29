import pickle
import os
from ambiguity_vectorizer import AmbiguityVectorizer
from data_loader_from_file import DataLoaderFromFile
from interpersonal_vectorizer import InterpersonalVectorizer
from incogurity_vectorizer import IncogurityVectorizer
import gensim

word2vec_pretrained = 'word2vec/GoogleNews-vectors-negative300.bin'
SHORT_RUN = True


if __name__ == "__main__":
    jokes_file = 'dataset/Jokes16000.txt'
    model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_pretrained,
                                                            binary=True, limit=50000)
    dataloader = DataLoaderFromFile(jokes_file)
    if SHORT_RUN:
        dataloader.all_sentences = dataloader.all_sentences[:100]
        dataloader.all_sentences_splitted_by_word = dataloader.all_sentences_splitted_by_word[:100]
    jokes_splitted_by_word = dataloader.get_all_sentences_splitted_by_word()
    #ambiguityVectorizer = AmbiguityVectorizer(jokes_splitted_by_word)
    interpersonal_vectorizer = InterpersonalVectorizer()
    interpersonal_features = interpersonal_vectorizer.get_feature_vector(sentences=jokes_splitted_by_word)
    jokes_splitted_by_word = dataloader.get_all_sentences_splitted_by_word()
    incogurityVector = IncogurityVectorizer(dataloader.all_sentences_no_repetition, model.similarity).vector
    pass

