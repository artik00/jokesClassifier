
WILSON_LEXICON = 'WilsonLexicon/subjclueslen1-HLTEMNLP05.tff'


def load_wilson_lexicon(lexicon_location=WILSON_LEXICON):
    """
    This loads the wilson subjectivity lexicon
    :param lexicon_location:
    :return:
    """
    strong_subjectivity_words_set = set()
    weak_subjectivity_words_set = set()
    negative_polarity_words_set = set()
    positive_polarity_words_set = set()
    line_key_to_value_dict = {}
    with open(lexicon_location, 'r') as file:
        for line in file:
            # each line is of the form "type=strongsubj len=1 word1=abuse pos1=verb stemmed1=y priorpolarity=negative"
            splitted_line = line.split()
            for term in splitted_line:
                key, value = term.partition('=')[::2]
                line_key_to_value_dict[key.strip()] = str(value)

            word = line_key_to_value_dict.get('word1')

            if line_key_to_value_dict.get('type') == 'strongsubj':
                strong_subjectivity_words_set.add(word)
            elif line_key_to_value_dict.get('type') == 'weaksubj':
                weak_subjectivity_words_set.add(word)

            if line_key_to_value_dict.get('priorpolarity') == 'negative':
                negative_polarity_words_set.add(word)
            elif line_key_to_value_dict.get('priorpolarity') == 'positive':
                positive_polarity_words_set.add(word)

    return strong_subjectivity_words_set, weak_subjectivity_words_set, negative_polarity_words_set,\
           positive_polarity_words_set
