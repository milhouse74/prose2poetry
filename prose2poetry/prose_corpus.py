import nltk
from nltk.corpus import gutenberg

nltk.download('gutenberg')


class ProseCorpus:
    default_gutenberg_prose_subset = ['austen-emma.txt', 'austen-persuasion.txt', 'austen-sense.txt']

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

            for gutenberg_fileid in gutenberg_fileids:
                corpus_lines = gutenberg.sents(gutenberg_fileid)

                # drop the first and last line which are usually the title and 'THE END' or 'FINIS'
                corpus_lines = corpus_lines[1:-1]

                chapter_lines = [c for c in corpus_lines if len(c) == 2 and c[0].lower() in ['volume', 'chapter']]

                # remove all CHAPTER/VOLUME markers
                corpus_lines = [c for c in corpus_lines if c not in chapter_lines]

                ##################################
                # apply other preprocessing here #
                ##################################

                # accumulate into a giant list
                self.sents.extend(corpus_lines)
