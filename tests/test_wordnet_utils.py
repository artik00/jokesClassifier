from utils.wordnet_utils import get_similar_words_from_wordnet, get_wn_tag_from_nltk_tag
from nltk.corpus import wordnet as wn
import pytest

class TestWordnetUtils:

    def test_tag_transator_positive(self):
        response = get_wn_tag_from_nltk_tag('NOUN')
        assert response == wn.NOUN

    def test_tag_transalor_negative(self):
        response = get_wn_tag_from_nltk_tag('OUN')
        assert response == None
