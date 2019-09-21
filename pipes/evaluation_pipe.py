from pipes.base_pipe import BasePipe
from data_loader_from_file import DataLoaderFromFile
import pickle
from itertools import combinations
from collections import defaultdict
import numpy as np
import statistics
import nltk

MAX_ANCHOR_LEN = 4
POS_FOR_ANCHORING = ["JJ", "JJR", "JJS", "NN", "NNS", "NNP",
                     "NNPS", "PDT", "PRP", "PRP$", "RB", "RBR", "RBS", "VB",
                     "VBD", "VBG", "VBN", "VBP", "VBZ"]


class EvaluationPipe(BasePipe):

    def __init__(self, path_to_sentence, max_number_of_sentences, path_to_model, path_to_anchors_file_1,
                 path_to_anchor_file_2, debug_mode=False):

        self.accuracy_list = list()
        self.sentence_number_to_anchors_dict = defaultdict(list)
        self.path_to_anchors = list()
        self.path_to_anchors.append(path_to_anchors_file_1)
        self.path_to_anchors.append(path_to_anchor_file_2)
        self.load_anchors()
        self.data_loader = DataLoaderFromFile(path_to_sentence, max_number_of_sentences=max_number_of_sentences)
        self.model = pickle.load(open(path_to_model, 'rb'))
        self.anchor_results = defaultdict(list)
        self.our_anchors_counter = 0
        self.missed_anchors_counter = 0
        self.debug_mode = debug_mode

    def load_anchors(self):
        """
        this function loads the given anchors extracted by the user to get evaluation score
        for the model anchoring , pre chosen anchors will be saved into self.sentence_number_to_anchors_dict
        :return:
        """

        for file_path in self.path_to_anchors:
            with open(file_path, 'r') as file:
                for line in file:
                    splitted_line = line.split()
                    sentence_line = 0
                    anchors = list()
                    for index, word in enumerate(splitted_line):
                        if index == 0:
                            sentence_line = word
                            pass
                        else:
                            anchors.append(word.lower())

                    if self.sentence_number_to_anchors_dict[sentence_line] and\
                            len(self.sentence_number_to_anchors_dict[sentence_line]) > 0:
                        temp = self.sentence_number_to_anchors_dict[sentence_line]
                        temp.append(anchors)
                    else:
                        temp = self.sentence_number_to_anchors_dict[sentence_line]
                        temp.append(anchors)

    def evaluate(self):
        """
        the function will evaluate the anchors in jokes
        :return:
        """
        for index, sentence in enumerate(self.data_loader.get_all_sentences_splitted_by_word()):
            if self.debug_mode:
                print(f"Sentence is {sentence}\n")
            sentence = self.remove_redundant_pos(sentence)
            if self.debug_mode:
                print(f"sentence after removing irrelevant POS IS {sentence}")

            possible_sentences_with_out_words = \
                self.get_all_possible_combinations_with_out_n_words(sentence, MAX_ANCHOR_LEN)

            vectors_of_possible_sentences = BasePipe.create_vector(self, possible_sentences_with_out_words)
            evaluation = self.model.predict(vectors_of_possible_sentences,
                                            batch_size=len(possible_sentences_with_out_words), verbose=1)

            minimum_index = np.argmin(evaluation)
            sentence_with_missing_words = possible_sentences_with_out_words[int(minimum_index)]

            anchor = [item for item in sentence if item not in sentence_with_missing_words]

            self.evaluate_anchor(anchor, index)
            if self.debug_mode:
                print(f"Number of sentence is {index}\n sentence after removing irrelevant POS IS {sentence}")

    def print_anchoring_score(self):
        """
        this function will write to the console the AOL(at-least-one) recall and precision of the model
        :return: None
        """
        total_anchor_hits = 0
        counter = 0
        hits = 0
        for index, list_of_results in self.anchor_results.items():
            # actual_anchors_list = self.sentence_number_to_anchors_dict[str(index + 1)]
            # accuracy = float((list_of_results[1]/(MAX_ANCHOR_LEN-1))))
            if list_of_results[1] > 0 or list_of_results[3] > 0:
                hits += 1
            counter += 1
            total_anchor_hits += list_of_results[1]

        # self.accuracy_list.append(accuracy)
        if self.debug_mode:
            print(f"anchoring accuracy is {statistics.mean(self.accuracy_list)}")

        print(f"anchoring MDE ALO recall is {float(hits/counter)}")
        print(f"anchoring MDE ALO precision is "
              f"{float(self.our_anchors_counter/(self.missed_anchors_counter + self.our_anchors_counter))}")

    def evaluate_anchor(self, anchor_from_model, sentence_index):
        """
        :param anchor_from_model: the anchors that was returned by the model as list of strings
        :param sentence_index: (int) the index of the original sentence to be match with golder anchors
        :return: None
        """
        expected_anchors_list = self.sentence_number_to_anchors_dict[str(sentence_index+1)]
        # The dict entrances  will be of index_of_Sentence-> [['anchors'], hits, ['anchors'], hits]
        for expected_anchors in expected_anchors_list:
            # accuracy, recall, precission, f1 = self.calculate_alo_anchoring(expected_anchors, anchor_from_model)
            self.our_anchors_counter += len(expected_anchors)
            words_that_appear_in_actual_and_expected = [x for x in expected_anchors if x in anchor_from_model]

            self.missed_anchors_counter += \
                len([anchor for anchor in anchor_from_model if anchor not in expected_anchors])

            self.anchor_results[sentence_index].append(words_that_appear_in_actual_and_expected)
            self.anchor_results[sentence_index].append(len(words_that_appear_in_actual_and_expected))

    def get_all_possible_combinations_with_out_n_words(self, sentence, n):
        """
        the function will return all the combinations of sentence with out set of n words
        :param sentence: original sentence - list of string
        :param n: how many words to remove from the original sentence
        :return: list of tuples that represents the combinations
        """
        tpl_to_list = []
        if n - 1 < len(sentence):
            tpl_list = list(combinations(sentence, len(sentence) - (n-1)))
        else:
            tpl_list = list(combinations(sentence, len(sentence) - 1))
        for tpl in tpl_list:
            tpl_to_list.append(" ".join([word for word in tpl]))
        return tpl_to_list

    def remove_redundant_pos(self, sentence):
        """
        the function will filter all POS that are not in the global variable POS_FOR_ANCHORING
        :param sentence: joke as list of string
        :return: list of string with out the irrelevant POS
        """
        tags = nltk.pos_tag(sentence)
        filter_by_pos = [t[0] for t in tags if t[1] in POS_FOR_ANCHORING]
        return filter_by_pos
