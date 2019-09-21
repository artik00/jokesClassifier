import cmudict
import numpy as np


class PhoneticStyle:
    """
       counting the number of repetitive prefixes in each words and the number of repeating suffix rhymes
       :return vector in which for each sentence return the length max of rhymes and prefixes alliteration
    """

    def __init__(self, sentences):
        """
        :param sentences: sentence to vectorize with phonetic feature
        """
        self.vector = None
        self.sentences = sentences

    def get_features_vector(self):
        cmu_in_py_dict = cmudict.dict()
        phonetic_vectors = []
        for sentence in self.sentences:
            phonetic_vectorizer = VectorizeByPhonetic(sentence=sentence, cmu_dict=cmu_in_py_dict)
            phonetic_vectorizer.vectorize()
            phonetic_vectors.append((phonetic_vectorizer.alliteration_num, phonetic_vectorizer.max_alliteration_len,
                                     phonetic_vectorizer.rhyme_num, phonetic_vectorizer.max_rhyme_len))

        self.vector = np.vstack(phonetic_vectors)
        return self.vector


class VectorizeByPhonetic:

    def __init__(self, sentence, cmu_dict):

        self.vowels = "AEIOU"

        """
        :param sentence: original input to make phonetic vector from
        :param cmu_dict: copy for cmu dict for phonetic parsing
        """
        self.alliteration_num = 0
        self.max_alliteration_len = 0
        self.rhyme_num = 0
        self.max_rhyme_len = 0
        self.sentence = sentence
        self.cmu_dict = cmu_dict

    def vectorize(self):
        """
        :return: return len and max size for suffix rhymes and prefix alliteration
        """
        alliteration_strings = {}
        rhyme_strings = {}
        for word in self.sentence:
            if word in self.cmu_dict:
                word_syllables = self.cmu_dict[word]

                first_syllable_in_word = set()
                suffix_rhymes = set()
                for pronunciation in word_syllables:
                    first_syllable_in_word.add(pronunciation[0])
                    i = 0
                    for i, phoneme in reversed(list(enumerate(pronunciation))):
                        if phoneme[0] in self.vowels:
                            break
                    suffix_rhymes.add("".join(pronunciation[i]))
                for first_phoneme in first_syllable_in_word:
                    # iterate count for each possible first phoneme
                    alliteration_strings[first_phoneme] = alliteration_strings.get(first_phoneme, 0) + 1
                for end_rhyme in suffix_rhymes:
                    # count all rhymes
                    rhyme_strings[end_rhyme] = rhyme_strings.get(end_rhyme, 0) + 1

        # ignore all non repetitive prefixes
        alliteration_strings = {syllable: occurrences for syllable, occurrences
                                in alliteration_strings.items() if occurrences > 1}

        rhyme_strings = {rhyme: count for rhyme, count in rhyme_strings.items() if count > 1}
        # number of repetitive prefixes
        self.alliteration_num = len(alliteration_strings)
        # longest alliteration string
        self.max_alliteration_len = max(alliteration_strings.values()) if self.alliteration_num > 0 else 0

        # number of rhymes prefixes
        self.rhyme_num = len(rhyme_strings)
        # longest repetitive  rhyme
        self.max_rhyme_len = max(rhyme_strings.values()) if self.rhyme_num > 0 else 0
