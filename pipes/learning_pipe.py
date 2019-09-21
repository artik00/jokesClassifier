from data_loader_from_file import DataLoaderFromFile
from pipes.base_pipe import BasePipe


class LearningPipe(BasePipe):

    def __init__(self, data_path, max_number_of_sentences):
        """
        :param data_path: os path to the data
        :param max_number_of_sentences: for debug limit the number of extracted sentences
        """
        self.data_path = data_path
        self.max_number_of_sentences = max_number_of_sentences
        self.data_loader = DataLoaderFromFile(self.data_path, max_number_of_sentences=self.max_number_of_sentences)

    def create_vector(self):
        """
        :return: vector for each sentance that represent the for humor method we present in the paper
        """
        return BasePipe.create_vector(self, self.data_loader.get_all_sentences())
