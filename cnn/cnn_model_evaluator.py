import pickle
from itertools import combinations


class CnnEvaluator:

    def __init__(self, model_path, path_to_sentences_to_extract_anchor):

        self.model = pickle.load(open(model_path, 'rb'))
        self.path = path_to_sentences_to_extract_anchor


    def evaluate_sentence(self, sentence):

        possible_sentences = self.get_all_possible_combinations_with_out_n_words(sentence, 3)
        min_probability = 1
        for possible_sentence in possible_sentences:
            answer = self.model.predict(sentence, batch_size=1, verbose=1)
            if answer < min_probability:
                min_sentence = possible_sentence
                min_probability = answer

        anchor = [item for item in sentence if item not in min_sentence]
        print(anchor)
        return answer[0]

    def get_all_possible_combinations_with_out_n_words(self, sentence, n):
        return list(combinations(sentence, len(sentence) - n))