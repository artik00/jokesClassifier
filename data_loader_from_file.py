# TODO currently not in use - need to remove
import collections
import os

class DataLoaderFromFile:

    problematic_punctuations = [".", "!", ",", "-"]

    def __init__(self, filename):
        self.filename = filename
        self.all_sentences = []
        self.all_sentences_splitted_by_word = []
        self.all_sentences_no_repetition = []
        with open(filename, 'rb') as file:
            for line in file:
                self.all_sentences.append(line.decode(errors='ignore').strip())
        self.break_each_sentence_into_tokens()
        self.remove_repetition()

    def get_all_sentences(self):
        return self.all_sentences

    def get_all_sentences_splitted_by_word(self):
        return self.all_sentences_splitted_by_word

    def break_each_sentence_into_tokens(self):
        for sentence in self.all_sentences:
            words = sentence.split()
            new_sentence = []
            for word in words:
                # for comma w/o spaces
                fixed_word = self.syntax_fixer(word)
                new_sentence += fixed_word
            self.all_sentences_splitted_by_word.append(new_sentence)

    def remove_repetition(self):
        for sentence in self.all_sentences_splitted_by_word:
            sentance_counter = collections.Counter(sentence)
            self.all_sentences_no_repetition.append([key for key in sentance_counter.keys()])





    def syntax_fixer(self, word):
        """
        The function checks for irrelevant punctuation and split/ remove etc

        :param word: given word
        :return: if problematic punctuation in word handle , else return original value
        """
        words_after_split = []

        if self.is_any_punc_in_word(word):
            stack = [word]
            candidate = stack.pop(0)
            while candidate:
                if self.is_any_punc_in_word(word):
                    break_after_first_split = False

                    for punc in self.problematic_punctuations:
                        if not break_after_first_split and punc in candidate:
                            break_after_first_split = True
                            # the list comp is to remove empty elements in the spited sentence for example in "..."
                            for split_part in [part for part in candidate.split(punc) if part]:
                                if not self.is_any_punc_in_word(split_part):
                                    words_after_split.append(split_part)
                                else:
                                    stack.append(split_part)
                else:
                    words_after_split.append(candidate)

                if stack:
                    candidate = stack.pop(0)
                else:
                    candidate = False
            return words_after_split
        else:
            return [word]

    def is_any_punc_in_word(self, word):
        """
        check if punctuation in word
        :param word: given word
        :return: if yes true else false
        """
        for punc in self.problematic_punctuations:
            if punc in word:
                return True
        return False


