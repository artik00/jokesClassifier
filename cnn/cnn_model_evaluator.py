import pickle
from itertools import combinations


class CnnEvaluator:

    def __init__(self, model_path, path_to_sentences_to_extract_anchor):
        """
        :param model_path: os path to the save model
        :param path_to_sentences_to_extract_anchor: the sentences to the "golden" anchors sentences
        """
        self.model = pickle.load(open(model_path, 'rb'))
        self.path = path_to_sentences_to_extract_anchor

    def evaluate_sentence(self, sentence):
        """
        function will find anchors on the given sentence using the model
        :param sentence: string
        :return: anchor for joke
        """
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
        """
        function will return all available combination of the original sentences with out any n words
        :param sentence: string
        :param n: number of words that can be removed , in order to calculate the length of the returned combinations
        :return:
        """
        return list(combinations(sentence, len(sentence) - n))
