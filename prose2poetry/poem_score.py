from .rhyme_score import rhyme_score
from .semantic_similarity import tfidf_vector_similarity
import itertools
import pronouncing
import difflib
import torch
import nltk


def pairs(seq):
    i = iter(seq)
    prev = next(i)
    for item in i:
        yield prev, item
        prev = item


def _semsim(line1, line2):
    try:
        return tfidf_vector_similarity(line1, line2)
    except AttributeError:
        return tfidf_vector_similarity(" ".join(line1), " ".join(line2))


def poem_score(poem_lines):
    num_words = 0
    last_words = set()
    stress_strings = []
    all_poem_words = []

    for pl in poem_lines:
        try:
            pwords = pl.split()
        except AttributeError:
            pwords = pl

        num_words += len(pwords)
        last_words.add(pwords[-1])

        all_poem_words.extend(pwords)

        stress_string = ""
        for pword in pwords:
            try:
                stress_string += pronouncing.stresses(
                    pronouncing.phones_for_word(pword)[0]
                )
            except:
                pass

        stress_strings.append(stress_string)

    last_word_combinations = itertools.combinations(last_words, 2)

    last_word_rhyme_score = 0.0
    for l in last_word_combinations:
        last_word_rhyme_score += rhyme_score(l[0], l[1])

    stress_string_combinations = itertools.combinations(stress_strings, 2)

    stress_string_score = 0.0
    for s in stress_string_combinations:
        stress_string_score += difflib.SequenceMatcher(None, s[0], s[1]).ratio()

    # rhyme score weight
    rsw = 1.5
    # stress score weight
    ssw = 0.5
    # next sentence prediction score weight
    nsw = 0.25

    semantic_score = 0.0
    for p in pairs(poem_lines):
        tmp = _semsim(p[0], p[1])
        #print("tmp! {0}".format(tmp))
        semantic_score += tmp

    # normalize by poem length in words
    ret = (
        rsw * last_word_rhyme_score + ssw * stress_string_score + nsw * semantic_score
    ) / num_words

    return ret
