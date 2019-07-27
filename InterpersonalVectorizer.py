from utils.wilson_lexicon_utils import load_wilson_lexicon
class InterpersonalVectorizer:

    def __init__(self, sentences):
        strong_subjectivity_words_set = set()
        weak_subjectivity_words_set = set()
        negative_polarity_words_set = set()
        positive_polarity_words_set = set()
        strong_subjectivity_words_set, weak_subjectivity_words_set, negative_polarity_words_set, \
        positive_polarity_words_set = load_wilson_lexicon()
