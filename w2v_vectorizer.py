import numpy as np
import gensim


class W2vVectorizer:
    
    def __init__(self, model: gensim.models, sentences: list):
        """
        
        :param model: w2v  model function  
        :param sentences: sentences to be vectorize in list
        """
        self.model = model
        self.sentences = sentences
        
    def get_features(self):
        """
        :return: list of vectors for the given sentences in the class init function
        """
        feature_w2v = []
        for sentence in self.sentences:
            sum_vector = np.zeros(self.model.vector_size)
            for word in sentence:
                try:
                    score = self.model.get_vector(word)
                except KeyError as e:
                    score = np.zeros(self.model.vector_size)
                sum_vector = sum_vector + score
            avg_vector = sum_vector / len(sentence) if sentence else np.zeros(self.model.vector_size)
            feature_w2v.append(avg_vector)
        return np.vstack(feature_w2v)
