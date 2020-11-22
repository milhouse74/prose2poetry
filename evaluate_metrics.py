#!/usr/bin/env python

import sys
import itertools
from prose2poetry.fasttext_model import FasttextModel
from prose2poetry.corpora import ProseCorpus, GutenbergCouplets, PFCouplets, pairs
from prose2poetry.couplet_score import CoupletScorer
import argparse
import numpy
import random


def compute_and_print_stats(corpus_name, scores):
    print(
        "{0} stats:\n\tmean: {1}\n\tmedian: {2}\n\tstddev: {3}\n\tvar: {4}\n\tptp: {5}\n\t75th quantile: {6}\n\t95th quantile: {7}".format(
            corpus_name,
            numpy.mean(scores),
            numpy.median(scores),
            numpy.std(scores),
            numpy.var(scores),
            numpy.ptp(scores),
            numpy.quantile(scores, 0.75),
            numpy.quantile(scores, 0.95),
        )
    )


def main():
    parser = argparse.ArgumentParser(
        prog="prose2poetry",
        description="Evaluate baseline metrics",
    )

    couplet_baseline_1 = GutenbergCouplets()
    couplet_baseline_2 = PFCouplets()

    # use default prose corpus for poem scoring - gutenberg novels from Jane Austen
    prose_corpus = ProseCorpus()

    # select 1000 random samples from our various baselines
    couplets_1 = random.sample(couplet_baseline_1.couplets, 1000)
    references_1 = itertools.chain(*couplet_baseline_1.couplets)

    # use the gold standard (gutenberg poetry) to evaluate all poems
    couplet_scorer = CoupletScorer(references_1)

    couplets_2 = random.sample(couplet_baseline_2.couplets, 1000)

    # use random pairs of sentences from our prose corpus for "bad couplets"
    couplets_3 = random.sample(list(pairs(prose_corpus.joined_sents)), 1000)

    couplet_b1_scores = numpy.ndarray(shape=(len(couplets_1),), dtype=numpy.float64)
    couplet_b2_scores = numpy.ndarray(shape=(len(couplets_2),), dtype=numpy.float64)
    prose_b1_scores = numpy.ndarray(shape=(len(couplets_3),), dtype=numpy.float64)

    print(
        "\nStep 1: calculating scores for Gutenberg poetry couplets ({0} data points)\n".format(
            len(couplets_1)
        )
    )
    for i, couplet in enumerate(couplets_1):
        print("evaluating couplet {0}".format(i))
        couplet_b1_scores[i] = couplet_scorer(couplet)

    compute_and_print_stats("couplet baseline 1 (gutenberg poems)", couplet_b1_scores)

    print(
        "\nStep 2: calculating scores for PoetryFoundation couplets ({0} data points)\n".format(
            len(couplets_2)
        )
    )
    for i, couplet in enumerate(couplets_2):
        print("evaluating couplet {0}".format(i))
        couplet_b2_scores[i] = couplet_scorer(couplet)

    compute_and_print_stats("couplet baseline 2 (poetry foundation)", couplet_b2_scores)

    print(
        "\nStep 3: calculating scores for pairs of lines in prose corpus ({0} data points)\n".format(
            len(couplets_3)
        )
    )
    for i, couplet in enumerate(couplets_3):
        print("evaluating couplet {0} {1}".format(i, couplet))
        prose_b1_scores[i] = couplet_scorer(couplet)

    compute_and_print_stats("prose baseline 1", prose_b1_scores)

    return 0


if __name__ == "__main__":
    sys.exit(main())
