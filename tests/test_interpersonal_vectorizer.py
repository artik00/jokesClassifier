from interpersonal_vectorizer import InterpersonalVectorizer
import numpy as np
class TestInterpersonalVectorizer:

    def test_create_vector_from_sentence(self):
        interpersonal_vectorizer = InterpersonalVectorizer('WilsonLexicon/subjclueslen1-HLTEMNLP05.tff')
        sentence = [["bad", "good", "excellent"]]
        vector = interpersonal_vectorizer.get_feature_vector(sentences=sentence)
        expected = np.array([[2, 1, 1, 2]])
        np.testing.assert_array_equal(vector, expected)
