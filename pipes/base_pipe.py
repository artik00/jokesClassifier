from concurrent.futures import ThreadPoolExecutor
from ambiguity_vectorizer import AmbiguityVectorizer
from interpersonal_vectorizer import InterpersonalVectorizer
from incogurity_vectorizer import IncogurityVectorizer
from phonetic_style import PhoneticStyle
from w2v_vectorizer import W2vVectorizer
import numpy as np
import gensim
from text_utils import remove_repetition, break_each_sentence_into_tokens
from abc import ABC

word2vec_pretrained = 'word2vec/GoogleNews-vectors-negative300.bin.gz'


class BasePipe(ABC):

    gensim_model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_pretrained,
                                                                   binary=True, limit=50000)

    def create_vector(self, sentences):
        sentences_tokenized = break_each_sentence_into_tokens(sentences)
        sentences_w_o_repetition = remove_repetition(sentences)
        executor = ThreadPoolExecutor(5)
        futures_tasks = {}
        word_vectorizer = W2vVectorizer(model=self.gensim_model,
                                        sentences=sentences_tokenized)
        futures_tasks["word2vec_vector"] = executor.submit(word_vectorizer.get_features)

        phonetic_vectorizer = PhoneticStyle(sentences)
        futures_tasks["phonetic_vector"] = executor.submit(phonetic_vectorizer.get_features_vector)

        ambiguity_vectorizer = AmbiguityVectorizer(sentences_tokenized)
        futures_tasks["ambiguity_vector"] = executor.submit(ambiguity_vectorizer.get_features_vector)

        incogurity_vectorizer = IncogurityVectorizer(sentences_w_o_repetition,
                                                     self.gensim_model.similarity)

        futures_tasks["incogurity_vector"] = executor.submit(incogurity_vectorizer.get_features_vector)

        interpersonal_vectorizer = InterpersonalVectorizer()
        futures_tasks["interpersonal_vector"] = executor.submit(
            interpersonal_vectorizer.get_feature_vector, sentences_tokenized)

        for future, vector in futures_tasks.items():
            vector.result()

        vectorize_data = np.concatenate(
            (futures_tasks["phonetic_vector"].result(), futures_tasks["ambiguity_vector"].result(),
             futures_tasks["incogurity_vector"].result(), futures_tasks["interpersonal_vector"].result(),
             futures_tasks["word2vec_vector"].result()), axis=1)

        return vectorize_data
