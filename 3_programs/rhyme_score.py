#!/usr/bin/env python3

import nltk
from nltk.corpus import wordnet
from collections import OrderedDict, defaultdict
import pronouncing
import itertools

from nltk.corpus import gutenberg


def _load_wpsm_matrix(path):
    ordered_phonemes = []
    by_phoneme = {}

    abs_max_phoneme = 0.0

    with open(path) as f:
        for i, l in enumerate(f):
            if i == 0:
                # first line, empty list per possible phoneme
                for phoneme in l.split():
                    ordered_phonemes.append(phoneme)
            else:
                scores = l.split()
                this_phoneme = scores[0]
                by_phoneme[this_phoneme] = defaultdict(list)
                for j, score in enumerate(scores[1:]):
                    score = float(score)
                    by_phoneme[this_phoneme][ordered_phonemes[j]] = score
                    abs_max_phoneme = max(abs_max_phoneme, abs(score))

    return by_phoneme, abs_max_phoneme


phoneme_similarity_matrix, normalizer = _load_wpsm_matrix('../2_data/wpsm.txt')


def rhyme_score(word1, word2):
    # penalize identical words
    if word1 == word2:
        return 0.0

    # phonetic mapping
    all_phones_1 = pronouncing.phones_for_word(word1)
    all_phones_2 = pronouncing.phones_for_word(word2)

    synsets_1 = wordnet.synsets(word1)
    synsets_2 = wordnet.synsets(word2)

    permutations = itertools.product(all_phones_1, all_phones_2)

    ret = []

    best_pos_1 = None
    best_pos_2 = None

    for permutation in permutations:
        phones_1 = permutation[0]
        phones_2 = permutation[1]

        phones_seq1 = phones_1.split()
        reversed_phones_seq1 = phones_seq1[::-1]

        phones_seq2 = phones_2.split()
        reversed_phones_seq2 = phones_seq2[::-1]

        # get minimun phonetic mapping
        min_len_phone = min(len(phones_seq1), len(phones_seq2))

        # get maximum length of phonemes
        max_len_phone = max(len(phones_seq1), len(phones_seq2))

        # calculate score
        phoneme_score = 0.0
        for i in range(min_len_phone):
            phoneme_1 = reversed_phones_seq1[i]
            phoneme_2 = reversed_phones_seq2[i]

            # exact match, good
            if phoneme_1 == phoneme_2:
                phoneme_score += 2.0

            # look for substitutions
            else:
                # *1.0 for matching stress
                # *0.5 for mismatching stress
                phoneme_1_stress = 0.0
                phoneme_2_stress = 0.0

                if phoneme_1[-1].isdigit():
                    phoneme_1, phoneme_1_stress = phoneme_1[:-1], float(phoneme_1[-1])
                if phoneme_2[-1].isdigit():
                    phoneme_2, phoneme_2_stress = phoneme_2[:-1], float(phoneme_2[-1])

                tmp = 0.0

                # same phoneme, possibly different stress
                if phoneme_1 == phoneme_2:
                    tmp = 2.0
                else:
                    # look for potential substitutions
                    tmp = phoneme_similarity_matrix.get(phoneme_1, {}).get(phoneme_2, 0.0)/normalizer

                # stress comparison
                if phoneme_1_stress == phoneme_2_stress:
                    tmp *= 1.0
                else:
                    # half score if stressed differently
                    tmp *= 0.5

                phoneme_score += tmp

        # syllable counts
        syl1 = pronouncing.syllable_count(phones_1)
        syl2 = pronouncing.syllable_count(phones_2)

        # add 1 if the syllable count is the same
        if syl1 == syl2:
            phoneme_score += 1.0

        # normalize by maximum length (to prefer longer matches)
        ret.append((phoneme_score/max_len_phone, phones_1, phones_2))

    ret = sorted(ret, key=lambda x: x[0], reverse=True)

    # return phones, from which we can try to convert to IPA or somehow guess a synset or POS
    return ret[0] if ret else (0.0, None, None)


if __name__ == '__main__':
    pairs = [
        ('cow', 'bow'),
        ('project', 'eject'),
        ('defect', 'detect'),
        ('behavior', 'savior'),
        ('squirrel', 'quarrel'),
        ('alabaster', 'plaster'),
        ('dog', 'cat'),
        ('dog', 'god'),
        ('dog', 'slog'),
        ('dog', 'frog'),
        ('dog', 'demagogue'),
        ('soda', 'mantra'),
        ('soda', 'toga'),
        ('soda', 'rodeo'),
        ('soda', 'Bogota'),
        ('banjo', 'commando'),
        ('banjo', 'piano'),
        ('banjo', 'tango'),
        ('banjo', 'nacho'),
        ('project', 'logic'),
        ('project', 'detect'),
        ('permit', 'hermit'),
        ('permit', 'remit'),
    ]
    #words = ['leicht', 'ached', 'elect', 'irked', 'eked', 'eject', 'oct', 'act', 'object', 'abject']
    #pairs = itertools.combinations(words, 2)

    all_results = []
    for p in pairs:
        all_results.append((rhyme_score(p[0], p[1]), p[0], p[1]))

    all_results = sorted(all_results, key=lambda x: x[0], reverse=True)

    for a in all_results:
        print('rhyme_score of {0}, {1}: {2}'.format(a[1], a[2], a[0][0]))
