#!/usr/bin/env python

from rhyme_score import rhyme_score
import itertools
import pronouncing
import difflib
from transformers import BertTokenizer, BertForNextSentencePrediction
import torch


def pairs(seq):
    i = iter(seq)
    prev = next(i)
    for item in i:
        yield prev, item
        prev = item


BertTokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
BertModel = BertForNextSentencePrediction.from_pretrained('bert-base-uncased', return_dict=True)


def _bert_next_sentence_prediction(line1, line2):
    print(line1)
    print(line2)
    encoding = BertTokenizer(line1, line2, return_tensors='pt')

    outputs = BertModel(**encoding, labels=torch.LongTensor([1]))
    logits = outputs.logits
    print(logits)
    print(logits[0, 0] < logits[0, 1])


def poem_score(poem_lines):
    print('poem: {0}'.format(poem_lines))
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
    # bert next sentence prediction score weight
    bsw = 0.5

    for p in pairs(poem_lines):
        _bert_next_sentence_prediction(p[0], p[1])

    # normalize by poem length in words
    ret = (rsw*last_word_rhyme_score + ssw*stress_string_score)/num_words

    return ret


if __name__ == '__main__':
    print(poem_score([
        'I went to the zoo',
        'I went to the loo',
    ]))

    print(poem_score([
        'I took my money to the bank on 23rd street',
        'My money was safe and my life became neat',
    ]))

    print(poem_score([
        'I took my money to the bank on 23rd street',
        'My monkey was cake and cockroaches have radiation',
    ]))

    print(poem_score([
        'I went to the zoo',
        'I also took my cousin to the loo',
    ]))

    print(poem_score([
        'I went to the zoo',
        'I ate cake',
    ]))

    print(poem_score([
        'Look in thy glass, and tell the face thou viewest',
        'Now is the time that face should form another',
        'Whose fresh repair if now thou not renewest',
        'Thou dost beguile the world, unbless some mother',
        'For where is she so fair whose unear\'d womb',
        'But if thou live, remember\'d not to be',
        'Die single, and thine image dies with thee'
    ]))
