import nltk
from nltk.corpus import gutenberg
import pronouncing
import gzip
import json
import pathlib
import string
import pandas

nltk.download("gutenberg")
_data_path = pathlib.Path(__file__).parent.absolute().joinpath("../data")


def pairs(seq):
    i = iter(seq)
    prev = next(i)
    for item in i:
        yield prev, item
        prev = item


class ProseCorpus:
    default_gutenberg_prose_subset = ["austen-emma.txt"]

    def __init__(self, custom_gutenberg_fileids=None, custom_corpus=None):
        self.sents = None

        if custom_corpus:
            self.sents = custom_corpus

        else:
            # let's use a subset of the nltk gutenberg corpus which are prose
            gutenberg_fileids = ProseCorpus.default_gutenberg_prose_subset
            if custom_gutenberg_fileids is not None:
                gutenberg_fileids = custom_gutenberg_fileids

            self.sents = []
            self.joined_sents = []

            for gutenberg_fileid in gutenberg_fileids:
                corpus_lines = gutenberg.sents(gutenberg_fileid)

                # drop the first and last line which are usually the title and 'THE END' or 'FINIS'
                corpus_lines = corpus_lines[1:-1]

                chapter_lines = [
                    c
                    for c in corpus_lines
                    if len(c) == 2 and c[0].lower() in ["volume", "chapter"]
                ]

                # remove all CHAPTER/VOLUME markers
                for c in corpus_lines:
                    if c not in chapter_lines:
                        ##################################
                        # apply other preprocessing here #
                        ##################################

                        self.sents.append(c)
                        self.joined_sents.append(" ".join(c))


class GutenbergCouplets:
    data_path = _data_path.joinpath("gutenberg-poetry.ndjson.gz")

    def __init__(self):
        self.couplets = []

        all_lines = []
        for line in gzip.open(str(GutenbergCouplets.data_path)):
            all_lines.append(json.loads(line.strip()))

        for pairs_of_lines in pairs(all_lines):
            if pairs_of_lines[0]["gid"] != pairs_of_lines[1]["gid"]:
                # not from same poem
                continue

            line1 = pairs_of_lines[0]["s"]
            line2 = pairs_of_lines[1]["s"]

            # strip punctuation from last word
            if line1[-1] in string.punctuation:
                line1 = line1[:-1]

            if line2[-1] in string.punctuation:
                line2 = line2[:-1]

            # use pronouncingpy to judge couplets
            if line2.split()[-1] in pronouncing.rhymes(line1.split()[-1]):
                self.couplets.append((line1, line2))


class PFCouplets:
    data_path = _data_path.joinpath("PoetryFoundationData.csv")

    def __init__(self):
        self.couplets = []

        pf_csvs = pandas.read_csv(str(PFCouplets.data_path))
        pf_csvs.head()

        all_lines = []
        for poem in pf_csvs.itertuples():
            all_lines.extend(
                [pl.replace("\n", "").strip() for pl in poem.Poem.split("\r\r") if pl]
            )

        for pairs_of_lines in pairs(all_lines):
            line1 = pairs_of_lines[0]
            line2 = pairs_of_lines[1]

            print(line1)
            print(line2)

            # strip punctuation from last word
            if line1[-1] in string.punctuation:
                line1 = line1[:-1]

            if line2[-1] in string.punctuation:
                line2 = line2[:-1]

            # use pronouncingpy to judge couplets
            if line2.split()[-1] in pronouncing.rhymes(line1.split()[-1]):
                print("these lines are a couplet!\n\t{0}\n\t{1}".format(line1, line2))
                self.couplets.append((line1, line2))
