from concurrent.futures import ThreadPoolExecutor
from ambiguity_vectorizer import AmbiguityVectorizer
from data_loader_from_file import DataLoaderFromFile
from interpersonal_vectorizer import InterpersonalVectorizer
from incogurity_vectorizer import IncogurityVectorizer
from phonetic_style import PhoneticStyle
from w2v_vectorizer import W2vVectorizer
import numpy as np
import gensim


class LearningPipe:

    def __init__(self, data_path, w2v_model, max_number_of_sentences):
        self.data_path = data_path
        self.w2v_model = w2v_model
        self.max_number_of_sentences = max_number_of_sentences

    def get_vectorize_data(self):

        data_loader = DataLoaderFromFile(self.data_path, max_number_of_sentences=self.max_number_of_sentences)
        jokes_splitted_by_word = data_loader.get_all_sentences_splitted_by_word()

        executor = ThreadPoolExecutor(5)
        futures_tasks = {}
        word_vectorizer = W2vVectorizer(model=self.w2v_model, sentences=data_loader.get_all_sentences_splitted_by_word())
        futures_tasks["word2vec_vector"] = executor.submit(word_vectorizer.get_features)

        phonetic_vectorizer = PhoneticStyle(data_loader.get_all_sentences())
        futures_tasks["phonetic_vector"] = executor.submit(phonetic_vectorizer.get_features_vector)

        ambiguity_vectorizer = AmbiguityVectorizer(jokes_splitted_by_word)
        futures_tasks["ambiguity_vector"] = executor.submit(ambiguity_vectorizer.get_features_vector)

        incogurity_vectorizer = IncogurityVectorizer(data_loader.get_all_sentences_without_repetition(),
                                                                 self.w2v_model.similarity)

        futures_tasks["incogurity_vector"] = executor.submit(incogurity_vectorizer.get_features_vector)

        interpersonal_vectorizer = InterpersonalVectorizer()
        futures_tasks["interpersonal_vector"] = executor.submit(
            interpersonal_vectorizer.get_feature_vector, jokes_splitted_by_word)

        for future, vector in futures_tasks.items():
            vector.result()

        vectorize_data = np.concatenate(
            (futures_tasks["phonetic_vector"].result(), futures_tasks["ambiguity_vector"].result(),
             futures_tasks["incogurity_vector"].result(), futures_tasks["interpersonal_vector"].result(),
             futures_tasks["word2vec_vector"].result()), axis=1)

        return vectorize_data
