from itertools import combinations
import numpy


class IncogurityVectorizer:

    def __init__(self, sentences, similarity_func, debug_mode=None):
        self.similarity_func = similarity_func
        self.sentences = sentences
        self.vector = []
        self.debug_mode = debug_mode

    def get_features_vector(self):
        """
        function create  vector for the Incongruity feature for all sentences
        :return:
        """
        for sentence in self.sentences:
            pairs = self.generate_words_couple_for_sentance(sentence)
            similarity_scores = []

            for pair in pairs:
                try:
                    similarity_score = self.similarity_func(pair[0], pair[1])
                    similarity_scores.append(similarity_score)
                except KeyError as e:
                    if self.debug_mode:
                        print(e)
                    similarity_scores.append(0)
            max_similarity = 0 if not similarity_scores else max(similarity_scores)
            min_similarity = 0 if not similarity_scores else min(similarity_scores)
            self.vector.append((max_similarity, min_similarity))

        self.vector = numpy.vstack(self.vector)
        return self.vector

    def generate_words_couple_for_sentance(self, sentence):
        """
        create all pair combination for the given sentence
        :param sentence: list of string to create pairs combination with
        :return: list of all available pairs of words from the original sentence
        """
        return list(combinations(sentence, 2))
