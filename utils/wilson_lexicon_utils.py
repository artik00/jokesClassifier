
WILSON_LEXICON = 'WilsonLexicon/subjclueslen1-HLTEMNLP05.tff'



def load_wilson_lexicon():
    strong_subjectivity_words_set = set()
    weak_subjectivity_words_set = set()
    negative_polarity_words_set = set()
    positive_polarity_words_set = set()
    with open(WILSON_LEXICON, 'rb') as file:
        for line in file:
            # each line is of the form "type=strongsubj len=1 word1=abuse pos1=verb stemmed1=y priorpolarity=negative"
            line = line.split()
            subjectivity = line[0].split('=')[1]
            word = line[2].split('=')[1]
            polarity = line[5].split('=')[1]
            # exactly 2 lines, 5549 and 5550, had "stemmed1=n m". Had to manually edit file to get it to work

            if subjectivity == "strongsubj":
                strong_subjectivity_words_set.add(word)
            elif subjectivity == "weaksubj":
                weak_subjectivity_words_set.add(word)

            if polarity == "negative":
                negative_polarity_words_set.add(word)
            elif polarity == "positive":
                positive_polarity_words_set.add(word)

    return strong_subjectivity_words_set, weak_subjectivity_words_set, negative_polarity_words_set,\
           positive_polarity_words_set
