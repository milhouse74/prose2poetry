from collections import OrderedDict, defaultdict
import pronouncing
import itertools
import pathlib
import sys
import numpy


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


data_path = pathlib.Path(__file__).parent.absolute().joinpath("../data")
phoneme_similarity_matrix, normalizer = _load_wpsm_matrix(
    data_path.joinpath("wpsm.txt")
)


def rhyme_score(word1, word2):
    word1 = word1.lower()
    word2 = word2.lower()

    # penalize identical words
    if word1 == word2:
        return -5.0

    all_phones_1 = pronouncing.phones_for_word(word1)
    all_phones_2 = pronouncing.phones_for_word(word2)

    if not all_phones_1 or not all_phones_2:
        #print(
        #    "couldnt find phonetic pronunciation for words {0}, {1}".format(
        #        word1, word2
        #    ),
        #    file=sys.stderr,
        #)
        return -5.0

    # use top phonetic mapping
    try:
        phones_1 = all_phones_1[0]
        phones_2 = all_phones_2[0]
    except Exception as e:
        print(
            "couldnt select top phonetic pronunciation for words {0}, {1}: {2}".format(
                word1, word2, str(e)
            ),
            file=sys.stderr,
        )
        return -5.0

    # add a penalty for multiple possible pronunciations
    # since it's not simple to associate to synsets, best avoid them (but write it up in the report!)
    multiple_pronunciation_penalty = 5 * (
        (len(all_phones_1) - 1) + (len(all_phones_2) - 1)
    )

    phones_seq1 = phones_1.split()
    reversed_phones_seq1 = phones_seq1[::-1]

    phones_seq2 = phones_2.split()
    reversed_phones_seq2 = phones_seq2[::-1]

    # get minimun phonetic mapping
    min_len_phone = min(len(phones_seq1), len(phones_seq2))

    # get maximum length of phonemes
    max_len_phone = max(len(phones_seq1), len(phones_seq2))

    # weight 1.0 on the last phoneme, 0.1 on the first phoneme, and importance between
    phoneme_position_weights = numpy.linspace(1.0, 0.1, min_len_phone)

    # calculate score
    phoneme_score = 0.0

    # incorporate position to weight the last phoneme the strongest and then gradually reduce
    for i in range(min_len_phone):
        phoneme_1 = reversed_phones_seq1[i]
        phoneme_2 = reversed_phones_seq2[i]

        position_weight = phoneme_position_weights[i]

        # exact match, good
        if phoneme_1 == phoneme_2:
            phoneme_score += 2.0*position_weight

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
                tmp = (
                    phoneme_similarity_matrix.get(phoneme_1, {}).get(phoneme_2, 0.0)
                    / normalizer
                )

            # stress comparison
            if phoneme_1_stress == phoneme_2_stress:
                tmp *= 1.0
            else:
                # half score if stressed differently
                tmp *= 0.5

            phoneme_score += tmp*position_weight

    # syllable counts
    syl1 = pronouncing.syllable_count(phones_1)
    syl2 = pronouncing.syllable_count(phones_2)

    # add 1 if the syllable count is the same
    if syl1 == syl2:
        phoneme_score += 1.0

    # penalize for ambiguous pronunciations
    phoneme_score -= multiple_pronunciation_penalty

    # normalize by maximum length (to prefer longer matches)
    return phoneme_score / max_len_phone
