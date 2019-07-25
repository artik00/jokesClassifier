from nltk.corpus import wordnet as wn
from six import iteritems
from math import isinf

def get_similar_words_from_wordnet(word, pos):
    return wn.synsets(word, pos)

def get_similarity_from_pathes(synset1, synset2):
    dist_dict1 = synset1._shortest_hypernym_paths(False)
    dist_dict2 = synset2._shortest_hypernym_paths(False)

    inf = float('inf')
    path_distance = inf

    for synset, d1 in iteritems(dist_dict1):
        d2 = dist_dict2.get(synset, inf)
        path_distance = min(path_distance, d1 + d2)

    return None if isinf(path_distance) else path_distance


def get_wn_tag_from_nltk_tag(tag_from_nltk):
    if tag_from_nltk.startswith('J'):
        return wn.ADJ
    elif tag_from_nltk.startswith('N'):
        return wn.NOUN
    elif tag_from_nltk.startswith('R'):
       return wn.ADV
    elif tag_from_nltk.startswith('V'):
        return wn.VERB
    return None