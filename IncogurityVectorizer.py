from itertools import combinations
import numpy


# a = model.similarityilarity('in', 'for')

class IncogurityVectorizer:

    def __init__(self, sentences, similarity_func):
        self.similarity_func = similarity_func
        self.sentences = sentences
        self.vector = []
        for sent in sentences:
            pairs = self.generate_words_couple_for_sentance(sent)
            similarity_scores = []

            for pair in pairs:
                try:
                    similarity_score = self.similarity_func(pair[0], pair[1])
                    similarity_scores.append(similarity_score)
                except KeyError as e:
                    print(f"IncogurityVectorizer - word wasnt found {str(e)}, adding 0 as similarity score for {pair}")
                    similarity_scores.append(0)
            max_similarity = max(similarity_scores)
            min_similarity = max(similarity_scores)
            self.vector.append(Incogurity(similarity=max_similarity, diff=min_similarity))

        self.vector = numpy.vstack(self.vector)






    def generate_words_couple_for_sentance(self, sentence):
        pairs = []
        # for pair in itertools.product(sentence, repeat=2):
        return list(combinations(sentence, 2))


class Incogurity:

    def __init__(self, similarity, diff):
        self.similarity = similarity
        self.diff = diff