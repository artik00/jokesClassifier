import pickle
import os
from DataLoader import DataLoader
from AmbiguityVectorizer import AmbiguityVectorizer
from DataLoaderFromFile import DataLoaderFromFile

if __name__ == "__main__":
    jokes_file = 'dataset/jokes100.txt'
    dataloader = DataLoaderFromFile(jokes_file)
    jokes_splitted_by_word = dataloader.get_all_sentences_splitted_by_word()
    ambiguityVectorizer = AmbiguityVectorizer(jokes_splitted_by_word)

