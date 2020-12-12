from collections import OrderedDict, defaultdict
import pronouncing
import itertools
import pathlib
import sys
import numpy


def rhyme_score(word1, word2, penalize_short_word=True):
    # initilize weight of the different scores
    weight_score_phone_matching = 0.33
    weight_score_consecutive_phone = 0.34
    weight_score_syllable_count = 0.33

    # penalizing variables
    penalizing_factor = 6

    # lowercasing of the words
    word1 = word1.lower()
    word2 = word2.lower()

    # get phonetic mapping of each word
    all_phones_1 = pronouncing.phones_for_word(word1)
    all_phones_2 = pronouncing.phones_for_word(word2)

    # penalize identical words
    if word1 == word2:
        return 0.0

    # no phone in one (or both) of the words
    if not all_phones_1 or not all_phones_2:
        return 0.0

    # one word can have multiple pronunciation
    permutations = list(itertools.product(all_phones_1, all_phones_2))
    ret = [0.0] * len(permutations)

    # loop over all possible permutation and will keep the best permutation
    for idx, permutation in enumerate(permutations):
        ### 1. PHONEME SCORING ###
        # left to right phoneme
        phones_1 = permutation[0]
        phones_2 = permutation[1]

        # right to left phoneme
        phones_seq1 = phones_1.split()
        reversed_phones_seq1 = phones_seq1[::-1]
        phones_seq2 = phones_2.split()
        reversed_phones_seq2 = phones_seq2[::-1]

        # get minimun phonetic mapping
        min_len_phone = min(len(phones_seq1), len(phones_seq2))

        ### 1.a) PHONEME MATCHING ###
        M = sum(
            [1 for phone in phones_seq1 if phone in phones_seq2]
            + [1 for phone in phones_seq2 if phone in phones_seq1]
        )
        T = len(phones_seq1) + len(phones_seq2)
        score_phone_matching = M / T

        ### 2. CONS. PHONEME ###
        consecutive_phone = 0
        for i in range(min_len_phone):
            if reversed_phones_seq1[i] != reversed_phones_seq2[i]:
                break
            consecutive_phone += 1
        score_consecutive_phone = consecutive_phone / min_len_phone

        ### 3. SYLLABLE SCORING ###
        # syllable counts
        syl1_count = pronouncing.syllable_count(phones_1)
        syl2_count = pronouncing.syllable_count(phones_2)
        if max(syl1_count, syl2_count) != 1:
            score_syllable_count = (min(syl1_count, syl2_count) - 1) / (
                max(syl1_count, syl2_count) - 1
            )
        else:
            score_syllable_count = 1

        ### COMBINED SCORE ###
        rhyme_score = (
            weight_score_phone_matching * score_phone_matching
            + weight_score_consecutive_phone * score_consecutive_phone
            + weight_score_syllable_count * score_syllable_count
        )

        # Add factors to penalize shorter rhyming words
        if penalize_short_word:
            rhyme_score = rhyme_score * min(1, min_len_phone / penalizing_factor)

        ret[idx] = rhyme_score

    ret = sorted(ret, reverse=True)

    return ret[0]
