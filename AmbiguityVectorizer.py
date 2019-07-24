from nltk import pos_tag_sents
from nltk.corpus import wordnet as wn
from utils.wordnet_utils import get_similar_words_from_wordnet

class AmbiguityVectorizer:

    def __init__(self, sentences):
        pos_tagged_documents = get_post_tags_from_nltk(sentences)
        create_sense_for_each_sentence(pos_tagged_documents)


#We want to get the POS tag for sentence  the expected input is list of list of str
def get_post_tags_from_nltk(sentences):
    return pos_tag_sents(sentences)

def create_sense_for_each_sentence(pos_tagged_sentences):
    for pos_tagged_sentence in pos_tagged_sentences:
        words_in_sentence = set()
        for word, tag in pos_tagged_sentence:
            tag_for_wn = get_wn_tag_from_nltk_tag(tag)
            if tag_for_wn:
                similar_words_from_wordnet = get_similar_words_from_wordnet(word, tag_for_wn)
                if similar_words_from_wordnet and len(similar_words_from_wordnet) > 0:
                    sense_farmost, sense_closest = get_farmost_closest_paths_for_words(word, words_in_sentence, similar_words_from_wordnet)
            words_in_sentence.update(word)


def get_farmost_closest_paths_for_words(words, words_in_sentence, similar_words_from_wordnet) :
    return 1, 1

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




