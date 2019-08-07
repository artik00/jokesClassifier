import collections

problematic_punctuations = [".", "!", ",", "-"]


def break_each_sentence_into_tokens(sentences):
    """
    :param sentences: list of sentences
    :return: list of sentences tokenized
    """
    sentences_into_tokens = []
    for sentence in sentences:
        words = sentence.split()
        new_sentence = []
        for word in words:
            # for comma w/o spaces
            fixed_word = syntax_fixer(word)
            new_sentence += fixed_word
        sentences_into_tokens.append(new_sentence)
    return sentences_into_tokens


def remove_repetition(sentences):
    """
    :param sentences: list of tokenized sentences
    :return: list sentences w/o repetitions
    """
    sentences_w_o_rep = []
    for sentence in sentences:
        sentance_counter = collections.Counter(sentence)
        sentences_w_o_rep.append([key for key in sentance_counter.keys()])
    return sentences_w_o_rep


def syntax_fixer(word):
    """
    The function checks for irrelevant punctuation and split/ remove etc

    :param word: given word
    :return: if problematic punctuation in word handle , else return original value
    """
    words_after_split = []

    if is_any_punc_in_word(word):
        stack = [word]
        candidate = stack.pop(0)
        while candidate:
            if is_any_punc_in_word(word):
                break_after_first_split = False
                global problematic_punctuations
                for punc in problematic_punctuations:
                    if not break_after_first_split and punc in candidate:
                        break_after_first_split = True
                        # the list comp is to remove empty elements in the spited sentence for example in "..."
                        for split_part in [part for part in candidate.split(punc) if part]:
                            if not is_any_punc_in_word(split_part):
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


def is_any_punc_in_word(word):
    """
    check if punctuation in word
    :param word: given word
    :return: if yes true else false
    """
    global problematic_punctuations
    for punc in problematic_punctuations:
        if punc in word:
            return True
    return False
