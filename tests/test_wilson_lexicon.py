from utils.wilson_lexicon_utils import load_wilson_lexicon


class TestWilsonLexicon:

    def test_wilson_lexicon(self):
        wilson_lexicon = load_wilson_lexicon()
        assert wilson_lexicon
