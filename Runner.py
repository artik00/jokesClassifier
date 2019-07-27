import pickle
import os
from data_loader_from_git import DataLoader
from ambiguity_vectorizer import AmbiguityVectorizer
from data_loader_from_file import DataLoaderFromFile

if __name__ == "__main__":
    jokes_file = 'dataset/jokes100.txt'
    dataloader = DataLoaderFromFile(jokes_file)
    jokes_splitted_by_word = dataloader.get_all_sentences_splitted_by_word()
    ambiguityVectorizerFeatures = AmbiguityVectorizer(jokes_splitted_by_word)

