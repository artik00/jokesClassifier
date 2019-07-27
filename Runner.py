import pickle
import os
from DataLoader import DataLoader
from AmbiguityVectorizer import AmbiguityVectorizer
from DataLoaderFromFile import DataLoaderFromFile
from interpersonal_vectorizer import InterpersonalVectorizer

if __name__ == "__main__":
    jokes_file = 'dataset/jokes100.txt'
    dataloader = DataLoaderFromFile(jokes_file)
    jokes_splitted_by_word = dataloader.get_all_sentences_splitted_by_word()
    #ambiguityVectorizer = AmbiguityVectorizer(jokes_splitted_by_word)
    interpersonal_vectorizer = InterpersonalVectorizer(jokes_splitted_by_word)

