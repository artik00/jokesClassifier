from nltk import pos_tag_sents
from math import log
from utils.wordnet_utils import get_similar_words_from_wordnet, get_similarity_from_pathes, get_wn_tag_from_nltk_tag
import numpy as np

class AmbiguityVectorizer:
    """
    This class received a list of list of strings where each sentence is breaken down to words.
    It goes over all the words in the sentence and extracts similar words from wordnet, wordnet returns a sysnet ,
    meaning a path of similar words. Next step is to go over all the word pairs in words , we do this only for NOUN,
    VERB, ADJ and ADVERB, we check the pathes similarity and get the maximum similar path and minimum path

    """
    def __init__(self, sentences):
        self.pos_tagged_documents = get_post_tags_from_nltk(sentences)

    def get_features_vector(self):
        return create_sense_for_each_sentence(pos_tagged_documents)



#We want to get the POS tag for sentence  the expected input is list of list of str
def get_post_tags_from_nltk(sentences):
    """
    :param sentences - list of list of strings:
    :return: POS tagged words from NLTK package
    """
    return pos_tag_sents(sentences)

def create_sense_for_each_sentence(pos_tagged_sentences):
    """

    :param pos_tagged_sentences: list of list of pairs , pair of (str, str) = (word, tag)
    :return: Stack of 3 scores for each sentence , (log(product of all pathes in sentence),
    minimum path distance, maximum path distance)
    """
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
    """

    :param list_of_similar_words_from_wordnet: sysnet structure from wordnet
    :param pathes_for_words_in_sentence: all other sysnet structures that we already encountered in the sentences before
    :return: (max similarity score , min similarity score) for sentece
    """
    max_path_similarity = 0
    min_path_similarity = 1
    for path in pathes_for_words_in_sentence:
        for symilarity_net in list_of_similar_words_from_wordnet:
            if path != symilarity_net:
                similarity = get_similarity_from_pathes(symilarity_net, path)
                max_path_similarity = max(max_path_similarity, similarity) if similarity else max_path_similarity
                min_path_similarity = min(min_path_similarity, similarity) if similarity else min_path_similarity
    return max_path_similarity, min_path_similarity







