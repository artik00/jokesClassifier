from ambiguity_vectorizer import get_post_tags_from_nltk
from utils.wordnet_utils import get_similar_words_from_wordnet, get_wn_tag_from_nltk_tag

class TestAmbiguityVectorizer:

    def test_pos_tagger(self):
        sentences = ['Jesus loves you but everyone else thinks you\'re an asshole'.split(),'A clear conscience is usually the sign of a bad memory'.split()]

        pos_sentences = get_post_tags_from_nltk(sentences)
        assert len(pos_sentences) == 2
        assert len(pos_sentences[0]) == 10
        assert pos_sentences[0][0][1] == 'NNP'


    def test_similarity_path(self):
        similar_words = get_similar_words_from_wordnet('world', get_wn_tag_from_nltk_tag('N'))
        print(similar_words)