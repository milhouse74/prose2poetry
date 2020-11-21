import nltk
from nltk.corpus import wordnet
import math
import sys
import numpy
from sklearn.feature_extraction.text import TfidfVectorizer
import re

nltk.download("wordnet")

_ALPHA = 0.2
_BETA = 0.45
_ETA = 0.4
_PHI = 0.2
_DELTA = 0.85


def _get_best_synset_pair(word_1, word_2):
    max_sim = -1.0
    synsets_1 = wordnet.synsets(word_1)
    synsets_2 = wordnet.synsets(word_2)
    if len(synsets_1) == 0 or len(synsets_2) == 0:
        return None, None
    else:
        max_sim = -1.0
        best_pair = None, None
        for synset_1 in synsets_1:
            for synset_2 in synsets_2:
                sim = wordnet.path_similarity(synset_1, synset_2)
                if sim != None and sim > max_sim:
                    max_sim = sim
                    best_pair = synset_1, synset_2
        return best_pair


def _length_dist(synset_1, synset_2):
    l_dist = sys.maxsize
    if synset_1 is None or synset_2 is None:
        return 0.0
    if synset_1 == synset_2:
        # if synset_1 and synset_2 are the same synset return 0
        l_dist = 0.0
    else:
        wset_1 = set([str(x.name()) for x in synset_1.lemmas()])
        wset_2 = set([str(x.name()) for x in synset_2.lemmas()])
        if len(wset_1.intersection(wset_2)) > 0:
            # if synset_1 != synset_2 but there is word overlap, return 1.0
            l_dist = 1.0
        else:
            # just compute the shortest path between the two
            l_dist = synset_1.shortest_path_distance(synset_2)
            if l_dist is None:
                l_dist = 0.0
    # normalize path length to the range [0,1]
    return math.exp(-_ALPHA * l_dist)


def _hierarchy_dist(synset_1, synset_2):
    h_dist = sys.maxsize
    if synset_1 is None or synset_2 is None:
        return h_dist
    if synset_1 == synset_2:
        # return the depth of one of synset_1 or synset_2
        h_dist = max([x[1] for x in synset_1.hypernym_distances()])
    else:
        # find the max depth of least common subsumer
        hypernyms_1 = {x[0]: x[1] for x in synset_1.hypernym_distances()}
        hypernyms_2 = {x[0]: x[1] for x in synset_2.hypernym_distances()}
        lcs_candidates = set(hypernyms_1.keys()).intersection(set(hypernyms_2.keys()))
        if len(lcs_candidates) > 0:
            lcs_dists = []
            for lcs_candidate in lcs_candidates:
                lcs_d1 = 0
                if lcs_candidate in hypernyms_1:
                    lcs_d1 = hypernyms_1[lcs_candidate]
                lcs_d2 = 0
                if lcs_candidate in hypernyms_2:
                    lcs_d2 = hypernyms_2[lcs_candidate]
                lcs_dists.append(max([lcs_d1, lcs_d2]))
            h_dist = max(lcs_dists)
        else:
            h_dist = 0
    return (math.exp(_BETA * h_dist) - math.exp(-_BETA * h_dist)) / (
        math.exp(_BETA * h_dist) + math.exp(-_BETA * h_dist)
    )


def _word_similarity(word_1, word_2):
    synset_pair = _get_best_synset_pair(word_1, word_2)
    return _length_dist(synset_pair[0], synset_pair[1]) * _hierarchy_dist(
        synset_pair[0], synset_pair[1]
    )


def _most_similar_word(word, word_set):
    max_sim = -1.0
    sim_word = ""
    for ref_word in word_set:
        sim = _word_similarity(word, ref_word)
        if sim > max_sim:
            max_sim = sim
            sim_word = ref_word
    return sim_word, max_sim

def _word_order_vector(words, joint_words, windex):
    wovec = numpy.zeros(len(joint_words))
    i = 0
    wordset = set(words)
    for joint_word in joint_words:
        if joint_word in wordset:
            # word in joint_words found in sentence, just populate the index
            wovec[i] = windex[joint_word]
        else:
            # word not in joint_words, find most similar word and populate
            # word_vector with the thresholded similarity
            sim_word, max_sim = _most_similar_word(joint_word, wordset)
            if max_sim > _ETA:
                wovec[i] = windex[sim_word]
            else:
                wovec[i] = 0
        i = i + 1
    return wovec

def _word_order_similarity(sentence_1, sentence_2):
    words_1 = nltk.word_tokenize(sentence_1)
    words_2 = nltk.word_tokenize(sentence_2)
    joint_words = list(set(words_1).union(set(words_2)))
    windex = {x[1]: x[0] for x in enumerate(joint_words)}
    r1 = _word_order_vector(words_1, joint_words, windex)
    r2 = _word_order_vector(words_2, joint_words, windex)
    return 1.0 - (numpy.linalg.norm(r1 - r2) / numpy.linalg.norm(r1 + r2))

class SemanticSimilarity:
    def __init__(self, prose_corpus, info_content_norm=False):
        self.corpus_freqs = dict()
        self.N = 0
        self.info_content_norm = info_content_norm

        for sent in prose_corpus.sents:
            for word in sent:
                word = word.lower()
                if word not in self.corpus_freqs:
                    self.corpus_freqs[word] = 0
                self.corpus_freqs[word] += 1
                self.N += 1

    def _info_content(self, lookup_word):
        lookup_word = lookup_word.lower()
        n = 0 if lookup_word not in self.corpus_freqs else self.corpus_freqs[lookup_word]
        return 1.0 - (math.log(n + 1) / math.log(N + 1))

    def _semantic_vector(self, words, joint_words):
        sent_set = set(words)
        semvec = numpy.zeros(len(joint_words))
        i = 0
        for joint_word in joint_words:
            if joint_word in sent_set:
                # if word in union exists in the sentence, s(i) = 1 (unnormalized)
                semvec[i] = 1.0
                if self.info_content_norm:
                    semvec[i] = semvec[i] * math.pow(self.info_content(joint_word), 2)
            else:
                # find the most similar word in the joint set and set the sim value
                sim_word, max_sim = _most_similar_word(joint_word, sent_set)
                semvec[i] = _PHI if max_sim > _PHI else 0.0
                if self.info_content_norm:
                    semvec[i] = (
                        semvec[i] * self.info_content(joint_word) * self.info_content(sim_word)
                    )
            i = i + 1
        return semvec
    
    def _semantic_similarity(self, sentence_1, sentence_2):
        words_1 = nltk.word_tokenize(sentence_1)
        words_2 = nltk.word_tokenize(sentence_2)
        joint_words = set(words_1).union(set(words_2))
        vec_1 = self._semantic_vector(words_1, joint_words)
        vec_2 = self._semantic_vector(words_2, joint_words)
        return numpy.dot(vec_1, vec_2.T) / (
            numpy.linalg.norm(vec_1) * numpy.linalg.norm(vec_2)
        )

    def similarity(self, sentence_1, sentence_2):
        return _DELTA * self._semantic_similarity(sentence_1, sentence_2) + (
            1.0 - _DELTA
        ) * _word_order_similarity(sentence_1, sentence_2)


# extra functions
def tfidf_vector_similarity(sentence_1, sentence_2):
    corpus = [sentence_1, sentence_2]
    vectorizer = TfidfVectorizer(min_df=1)
    vec_1 = vectorizer.fit_transform(corpus).toarray()[0]
    vec_2 = vectorizer.fit_transform(corpus).toarray()[1]
    sim = numpy.dot(vec_1, vec_2.T) / (
        numpy.linalg.norm(vec_1) * numpy.linalg.norm(vec_2)
    )
    return sim


def jaccard_similarity_coefficient(sentence_1, sentence_2):
    words_1 = nltk.word_tokenize(sentence_1)
    words_2 = nltk.word_tokenize(sentence_2)
    joint_words = set(words_1).union(set(words_2))
    intersection_words = set(words_1).intersection(set(words_2))
    return len(intersection_words) / len(joint_words)
