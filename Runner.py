from pipes.learning_pipe import LearningPipe
from pipes.evaluation_pipe import EvaluationPipe
from cnn.cnn import CNN
import numpy as np

SHORT_RUN = True
NUMBER_OF_SENTENCES = 16000
NUMBER_OF_SENTENCES_FOR_ANCHORING = 200


if __name__ == "__main__":
    jokes_file = 'dataset/Jokes16000.txt'
    non_jokes_file = 'dataset/MIX16000.txt'

    # model build

    jokes_total_data_instnce = LearningPipe(data_path=jokes_file,
                                            max_number_of_sentences=NUMBER_OF_SENTENCES)
    non_jokes_data_instance = LearningPipe(data_path=non_jokes_file,
                                           max_number_of_sentences=NUMBER_OF_SENTENCES)
    jokes_total_vector = jokes_total_data_instnce.create_vector()
    non_jokes_total_vector = non_jokes_data_instance.create_vector()

    labels = [0] * NUMBER_OF_SENTENCES + [1] * NUMBER_OF_SENTENCES
    labels = np.asarray(labels)
    concatenated_data = np.concatenate([jokes_total_vector, non_jokes_total_vector])

    cnn_instance = CNN(data=concatenated_data, labels=labels)
    cnn_model = cnn_instance.model

    # for paper needs
    # print(f"Model precision - {cnn_instance.precision}")
    # print(f"Model recall - {cnn_instance.recall}")
    # print(f"Model F1 score - {cnn_instance.f1_score}")
    # print(f"Model Accuracy - {cnn_instance.accuracy}")
    # print(f"Model loss - {cnn_instance.loss}")

    # enable to get anchor accuracy
    eval_pipe = EvaluationPipe('dataset/sentences_for_anchoring.txt', NUMBER_OF_SENTENCES_FOR_ANCHORING,
                               'model/model_first.sav', 'dataset/amir_anchors.txt', 'dataset/artyom_anchors.txt')
    eval_pipe.evaluate()

    eval_pipe.print_anchoring_score()
