from pipes.base_pipe import BasePipe
from data_loader_from_file import DataLoaderFromFile
import pickle
from itertools import combinations
import numpy as np

class EvaluationPipe(BasePipe):

    def __init__(self, path_to_sentence, max_number_of_sentences, path_to_model):
        self.data_loader = DataLoaderFromFile(path_to_sentence, max_number_of_sentences=max_number_of_sentences)
        self.model = pickle.load(open(path_to_model, 'rb'))

    def evaluate(self):
        for sentence in self.data_loader.get_all_sentences_splitted_by_word():
            possible_sentences_with_out_words = self.get_all_possible_combinations_with_out_n_words(sentence, 4)
            vectors_of_possible_sentences = BasePipe.create_vector(self, possible_sentences_with_out_words)
            evaluation = self.model.predict(vectors_of_possible_sentences,
                                            batch_size=len(possible_sentences_with_out_words), verbose=1)

            minimum_index = np.argmin(evaluation)
            sentence_with_missing_words = possible_sentences_with_out_words[minimum_index]

            anchor = [item for item in sentence if item not in sentence_with_missing_words]

            print(sentence_with_missing_words)
            print(sentence)
            print(anchor)

    def get_all_possible_combinations_with_out_n_words(self, sentence, n):
        tpl_to_list = []
        tpl_list = list(combinations(sentence, len(sentence) - n))
        for tpl in tpl_list:
            tpl_to_list.append(" ".join([word for word in tpl]))
        return tpl_to_list

