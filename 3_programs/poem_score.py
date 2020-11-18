#!/usr/bin/env python

from rhyme_score import rhyme_score
import itertools
import pronouncing
import difflib


def poem_score(poem_lines):
    # first, get all the rhyme score of all permutations/combinations of last words

    num_words = 0
    last_words = set()
    stress_strings = []
    all_poem_words = []

    for pl in poem_lines:
        pwords = pl.split()

        num_words += len(pwords)
        last_words.add(pwords[-1])

        all_poem_words.extend(pwords)

        stress_string = ''
        for pword in pwords:
            try:
                stress_string += pronouncing.stresses(pronouncing.phones_for_word(pword)[0])
            except:
                pass

        stress_strings.append(stress_string)

    last_word_combinations = itertools.combinations(last_words, 2)

    last_word_rhyme_score = 0.0
    for l in last_word_combinations:
        last_word_rhyme_score += rhyme_score(l[0], l[1])[0]

    stress_string_combinations = itertools.combinations(stress_strings, 2)

    stress_string_score = 0.0
    for s in stress_string_combinations:
        stress_string_score += difflib.SequenceMatcher(None, s[0], s[1]).ratio()

    # rhyme score weight
    rsw = 1.0
    # stress score weight
    ssw = 0.5

    # normalize by poem length in words
    ret = (rsw*last_word_rhyme_score+ssw*stress_string_score)/num_words

    return ret


if __name__ == '__main__':
    #print(poem_score([
    #    'I went to the zoo',
    #    'I went to the loo',
    #]))

    print(poem_score([
        'I took my money to the bank on 23rd street',
        'My money was safe and my life became neat',
    ]))

    print(poem_score([
        'I took my money to the bank on 23rd street',
        'My monkey was cake and cockroaches have radiation',
    ]))

    #print(poem_score([
    #    'I went to the zoo',
    #    'I also took my cousin to the loo',
    #]))

    #print(poem_score([
    #    'I went to the zoo',
    #    'I ate cake',
    #]))
