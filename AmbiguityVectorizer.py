from nltk import pos_tag_sents
from nltk.corpus import wordnet as wn
from math import log
from utils.wordnet_utils import get_similar_words_from_wordnet, get_similarity_from_pathes, get_wn_tag_from_nltk_tag
import numpy as np

class AmbiguityVectorizer:

    def __init__(self, sentences):
        pos_tagged_documents = get_post_tags_from_nltk(sentences)
        create_sense_for_each_sentence(pos_tagged_documents)


#We want to get the POS tag for sentence  the expected input is list of list of str
def get_post_tags_from_nltk(sentences):
    return pos_tag_sents(sentences)

def create_sense_for_each_sentence(pos_tagged_sentences):
    feature_vectors = []
    for pos_tagged_sentence in pos_tagged_sentences:
        pathes_for_words_in_sentence = set()
        sense_combination_product = 1
        max_sense_farmost = 0
        min_sense_closest = 100
        for word, tag in pos_tagged_sentence:
            tag_for_wn = get_wn_tag_from_nltk_tag(tag)
            if tag_for_wn:
                list_of_similar_words_from_wordnet = get_similar_words_from_wordnet(word, tag_for_wn)

                if list_of_similar_words_from_wordnet and len(list_of_similar_words_from_wordnet) > 0:
                    sense_combination_product *= len(list_of_similar_words_from_wordnet)
                    sense_farmost, sense_closest = get_farmost_and_closest_paths_for_pathes(list_of_similar_words_from_wordnet, pathes_for_words_in_sentence)
                    pathes_for_words_in_sentence.update(list_of_similar_words_from_wordnet)

                    min_sense_closest = min(min_sense_closest, sense_closest)
                    max_sense_farmost = max(max_sense_farmost, sense_farmost)
        log_of_sense_combination_product = log(sense_combination_product)
        feature_vectors.append((log_of_sense_combination_product, max_sense_farmost, 0 if min_sense_closest == 100 else min_sense_closest))
    return np.vstack(feature_vectors)


def get_farmost_and_closest_paths_for_pathes(list_of_similar_words_from_wordnet, pathes_for_words_in_sentence):
    max_path_similarity = 0
    min_path_similarity = 1
    for path in pathes_for_words_in_sentence:
        for symilarity_net in list_of_similar_words_from_wordnet:
            similarity = get_similarity_from_pathes(symilarity_net, path)
            max_path_similarity = max(max_path_similarity, similarity) if similarity else max_path_similarity
            min_path_similarity = min(min_path_similarity, similarity) if similarity else min_path_similarity
    return max_path_similarity, min_path_similarity







