import itertools
from collections import defaultdict
import pronouncing


class NaiveGenerator:
    def __init__(self, prose_corpus):
        # iterate through the prose corpus
        # find all last words that rhyme, collect these into "couplets"

        self.last_words = defaultdict(list)
        for sent in prose_corpus.sents:
            if not sent:
                continue

            # accumulate all sentences that end with the same last word
            self.last_words[sent[-1]].append(sent)

        last_word_pairs = list(itertools.combinations(self.last_words.keys(), 2))

        self.rhymes = []

        for i, word_pair in enumerate(last_word_pairs):
            if word_pair[1] in pronouncing.rhymes(word_pair[0]):
                self.rhymes.append(word_pair)

    def generate_couplets(self, n=10):
        ret = []
        for pair in self.rhymes[:n]:
            ret.append(
                [
                    # pick the shortest candidates
                    " ".join(min(self.last_words[pair[0]], key=len)),
                    " ".join(min(self.last_words[pair[1]], key=len)),
                ]
            )
        return ret
