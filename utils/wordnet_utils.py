from nltk.corpus import wordnet as wn


def get_similar_words_from_wordnet(word, pos):
    return wn.synsets(word, pos)