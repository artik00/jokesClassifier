from nltk.corpus import wordnet as wn
from six import iteritems
from math import isinf


def get_similar_words_from_wordnet(word, pos):
    """

    :param word: word to get path for
    :param pos: tag of th word
    :return: path of similar words for gived word and tag
    """
    return wn.synsets(word, pos)

def get_similarity_from_pathes(synset1, synset2):
    """

    :param synset1:  Path of similar words
    :param synset2: 2nd Path of similar words
    :return: distance between 2 pathes
    """
    dist_dict1 = synset1._shortest_hypernym_paths(False)
    dist_dict2 = synset2._shortest_hypernym_paths(False)

    inf = float('inf')
    path_distance = inf

    for synset, d1 in iteritems(dist_dict1):
        d2 = dist_dict2.get(synset, inf)
        path_distance = min(path_distance, d1 + d2)

    return None if isinf(path_distance) else path_distance


def get_wn_tag_from_nltk_tag(tag_from_nltk):
    """

    :param tag_from_nltk: POS tag from nltk
    :return: a tag according to wordnet corpus
    """
    transaltion_dict = {'J': wn.ADJ, 'N': wn.NOUN, 'R': wn.ADV, 'V': wn.VERB}
    return transaltion_dict.get(tag_from_nltk[0])