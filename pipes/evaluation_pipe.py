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

    def __init__(self, path_to_sentence, max_number_of_sentences, path_to_model, path_to_anchors_file_1, path_to_anchor_file_2):
        self.path_to_anchors = list()
        self.path_to_anchors.append(path_to_anchors_file_1)
        self.path_to_anchors.append(path_to_anchor_file_2)
        self.load_anchors()
        self.data_loader = DataLoaderFromFile(path_to_sentence, max_number_of_sentences=max_number_of_sentences)
        self.model = pickle.load(open(path_to_model, 'rb'))
        self.anchor_results = defaultdict(list)
        self.our_anchors_counter = 0
        self.missed_anchors_counter = 0


    def load_anchors(self):
        self.sentence_number_to_anchors_dict = defaultdict(list)
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

                    if self.sentence_number_to_anchors_dict[sentence_line] and len(self.sentence_number_to_anchors_dict[sentence_line]) > 0:
                        temp = self.sentence_number_to_anchors_dict[sentence_line]
                        temp.append(anchors)
                    else:
                        temp = self.sentence_number_to_anchors_dict[sentence_line]
                        temp.append(anchors)


    def evaluate(self):
        self.accuracy_list = list()
        for index, sentence in enumerate(self.data_loader.get_all_sentences_splitted_by_word()):
            # print(f"Sentence is {sentence}\n")
            sentence = self.remove_redundant_pos(sentence)
            # print(f"sentence after removing irrelevant POS IS {sentence}")

            possible_sentences_with_out_words = self.get_all_possible_combinations_with_out_n_words(sentence, MAX_ANCHOR_LEN)
            vectors_of_possible_sentences = BasePipe.create_vector(self, possible_sentences_with_out_words)
            evaluation = self.model.predict(vectors_of_possible_sentences,
                                            batch_size=len(possible_sentences_with_out_words), verbose=1)

            minimum_index = np.argmin(evaluation)
            sentence_with_missing_words = possible_sentences_with_out_words[minimum_index]

            anchor = [item for item in sentence if item not in sentence_with_missing_words]

            self.evaluate_anchor(anchor, index)

            print(f"Number of sentence is {index}\n sentence after removing irrelevant POS IS {sentence}")

    def print_anchoring_score(self):
        total_anchor_hits = 0
        counter = 0
        hits = 0
        for index, list_of_results in self.anchor_results.items():
            # actual_anchors_list = self.sentence_number_to_anchors_dict[str(index + 1)]
            #accuracy = float((list_of_results[1]/(MAX_ANCHOR_LEN-1)/2)) + \
            #           float((list_of_results[3]/(MAX_ANCHOR_LEN-1)/2))
            if list_of_results[1] > 0 or list_of_results[3] > 0:
                hits += 1
            counter += 1
            total_anchor_hits += list_of_results[1]

        #self.accuracy_list.append(accuracy)
        #print(f"anchoring accuracy is {statistics.mean(self.accuracy_list)}")
        print(f"anchoring MDE ALO recall is {float(hits/counter)}")
        print(f"anchoring MDE ALO precision is "
              f"{float(self.our_anchors_counter/(self.missed_anchors_counter + self.our_anchors_counter))}")



    def evaluate_anchor(self, anchor_from_model, sentence_index):
        expected_anchors_list = self.sentence_number_to_anchors_dict[str(sentence_index+1)]
        # The fict will be of index_of_Sentence-> [['anchors'], hits, ['anchors'], hits]
        for expected_anchors in expected_anchors_list:
            #accuracy, recall, precission, f1 = self.calculate_alo_anchoring(expected_anchors, anchor_from_model)
            self.our_anchors_counter += len(expected_anchors)
            words_that_appear_in_actual_and_expected = [x for x in expected_anchors if x in anchor_from_model]
            self.missed_anchors_counter += len([anchor for anchor in anchor_from_model if anchor not in expected_anchors])
            self.anchor_results[sentence_index].append(words_that_appear_in_actual_and_expected)
            self.anchor_results[sentence_index].append(len(words_that_appear_in_actual_and_expected))


    def get_all_possible_combinations_with_out_n_words(self, sentence, n):
        tpl_to_list = []
        if n - 1 < len(sentence):
            tpl_list = list(combinations(sentence, len(sentence) - (n-1)))
        else :
            tpl_list = list(combinations(sentence, len(sentence) - 1))
        for tpl in tpl_list:
            tpl_to_list.append(" ".join([word for word in tpl]))
        return tpl_to_list

    def remove_redundant_pos(self, sentence):
        tags = nltk.pos_tag(sentence)
        filter_by_pos = [t[0] for t in tags if t[1] in POS_FOR_ANCHORING]
        return filter_by_pos
