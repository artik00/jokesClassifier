from utils.wilson_lexicon_utils import load_wilson_lexicon
import numpy as np

class InterpersonalVectorizer:
    """This class looks for subjectivity difference between all words
    """
    def __init__(self):
        self.strong_subjectivity_words_set, self.weak_subjectivity_words_set  = set(), set()
        self.negative_polarity_words_set, self.positive_polarity_words_set = set(), set()

        self.strong_subjectivity_words_set, self.weak_subjectivity_words_set, self.negative_polarity_words_set, \
        self.positive_polarity_words_set = load_wilson_lexicon()

    def get_feature_vector(self, sentences):
        strong_subjectivity_counter, weak_subjectivity_counter = 0, 0
        negative_polarity_counter, positive_polarity_counter = 0, 0
        feature_vector = []
        for sentence in sentences:
            for word in sentence:
                if word in self.strong_subjectivity_words_set:
                    strong_subjectivity_counter += 1
                if word in self.weak_subjectivity_words_set:
                    weak_subjectivity_counter += 1
                if word in self.negative_polarity_words_set:
                    negative_polarity_counter += 1
                if word in self.positive_polarity_words_set:
                    positive_polarity_counter += 1
            feature_vector.append((strong_subjectivity_counter, weak_subjectivity_counter,
                                   negative_polarity_counter, positive_polarity_counter))
            strong_subjectivity_counter, weak_subjectivity_counter = 0, 0
            negative_polarity_counter, positive_polarity_counter = 0, 0
        return np.vstack(feature_vector)

