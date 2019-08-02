from itertools import combinations
import numpy


class IncogurityVectorizer:

    def __init__(self, sentences, similarity_func):
        self.similarity_func = similarity_func
        self.sentences = sentences
        self.vector = []

    def get_features_vector(self):
        for sentence in self.sentences:
            pairs = self.generate_words_couple_for_sentance(sentence)
            similarity_scores = []

            for pair in pairs:
                try:
                    similarity_score = self.similarity_func(pair[0], pair[1])
                    similarity_scores.append(similarity_score)
                except KeyError as e:
                    #print(f"IncogurityVectorizer - word wasnt found {str(e)}, adding 0 as similarity score for {pair}")
                    similarity_scores.append(0)
            max_similarity = 0 if not similarity_scores else max(similarity_scores)
            min_similarity = 0 if not similarity_scores else min(similarity_scores)
            self.vector.append((max_similarity, min_similarity))

        self.vector = numpy.vstack(self.vector)
        return self.vector

    def generate_words_couple_for_sentance(self, sentence):
        return list(combinations(sentence, 2))

