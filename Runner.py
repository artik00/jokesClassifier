import gensim
from learning_pipe import LearningPipe
word2vec_pretrained = 'word2vec/GoogleNews-vectors-negative300.bin.gz'
SHORT_RUN = True


if __name__ == "__main__":
    jokes_file = 'dataset/Jokes16000.txt'
    non_jokes_file = 'dataset/MIX16000.txt'
    model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_pretrained,
                                                            binary=True, limit=50000)

    NUMBER_OF_SENTENCES = 20
    jokes_total_data_instnce = LearningPipe(data_path=jokes_file,  w2v_model=model,
                                            max_number_of_sentences=NUMBER_OF_SENTENCES)
    non_jokes_data_instance = LearningPipe(data_path=non_jokes_file,  w2v_model=model,
                                           max_number_of_sentences=NUMBER_OF_SENTENCES)
    jokes_total_vector = jokes_total_data_instnce.get_vectorize_data()
    non_jokes_total_vector = non_jokes_data_instance.get_vectorize_data()

    pass

