#!/usr/bin/env python

import sys
from prose2poetry.corpora import ProseCorpus, GutenbergCouplets, PFCouplets, pairs
from prose2poetry.couplet_score import CoupletScorer
from prose2poetry.generators import NaiveGenerator
import argparse
import numpy
import random


def compute_and_print_stats(corpus_name, scores):
    print(
        "{0} stats:\n\tmean: {1}\n\tmedian: {2}\n\tstddev: {3}\n\tvar: {4}\n\tptp: {5}\n\t75th quantile: {6}\n\t95th quantile: {7}".format(
            corpus_name,
            numpy.mean(scores, axis=0),
            numpy.median(scores, axis=0),
            numpy.std(scores, axis=0),
            numpy.var(scores, axis=0),
            numpy.ptp(scores, axis=0),
            numpy.quantile(scores, 0.75, axis=0),
            numpy.quantile(scores, 0.95, axis=0),
        )
    )


def main():
    parser = argparse.ArgumentParser(
        prog="prose2poetry",
        description="Evaluate baseline metrics",
    )

    parser.add_argument(
        "--n-eval",
        type=int,
        default=1000,
        help="Number of couplets to sample for evaluation from all corpora/generators",
    )

    parser.add_argument(
        "--rand-seed",
        type=int,
        default=42,
        help="Integer seed for rng",
    )

    args = parse_args()

    # set up random seed to replicate
    random.seed(args.rand_seed)

    couplet_baseline_1 = GutenbergCouplets()
    couplet_baseline_2 = PFCouplets()

    # select args.n_eval random samples from our various baselines
    couplets_1 = random.sample(couplet_baseline_1.couplets, args.n_eval)
    couplets_2 = random.sample(couplet_baseline_2.couplets, args.n_eval)

    # use a random selection of gold standard gutenberg couplets to evaluate all poems
    couplet_scorer = CoupletScorer(
        couplet_baseline_1.couplets_flat_list(n_random_couplets=5000)
    )

    couplet_b1_scores = numpy.ndarray(shape=(len(couplets_1), 5), dtype=numpy.float64)
    couplet_b2_scores = numpy.ndarray(shape=(len(couplets_2), 5), dtype=numpy.float64)

    print(
        "\nStep 1: calculating scores for Gutenberg poetry couplets ({0} data points)\n".format(
            len(couplets_1)
        )
    )
    for i, couplet in enumerate(couplets_1):
        # print("evaluating couplet {0}".format(i))
        couplet_b1_scores[i] = couplet_scorer.calculate_scores(couplet)

    compute_and_print_stats("couplet baseline 1 (gutenberg poems)", couplet_b1_scores)

    print(
        "\nStep 2: calculating scores for PoetryFoundation couplets ({0} data points)\n".format(
            len(couplets_2)
        )
    )
    for i, couplet in enumerate(couplets_2):
        # print("evaluating couplet {0}".format(i))
        couplet_b2_scores[i] = couplet_scorer.calculate_scores(couplet)

    compute_and_print_stats("couplet baseline 2 (poetry foundation)", couplet_b2_scores)

    ## use random pairs of sentences from our prose corpus for "bad couplets"
    prose_corpus = ProseCorpus()
    couplets_3 = random.sample(list(pairs(prose_corpus.joined_sents)), args.n_eval)
    prose_b1_scores = numpy.ndarray(shape=(len(couplets_3), 5), dtype=numpy.float64)

    print(
        "\nStep 3: calculating scores for pairs of lines in prose corpus ({0} data points)\n".format(
            len(couplets_3)
        )
    )
    for i, couplet in enumerate(couplets_3):
        # print("evaluating couplet {0}".format(i))
        prose_b1_scores[i] = couplet_scorer.calculate_scores(couplet)

    compute_and_print_stats("prose baseline 1", prose_b1_scores)

    gen = NaiveGenerator(prose_corpus)
    couplets_4 = gen.generate_couplets(n=args.n_eval)
    naive_scores = numpy.ndarray(shape=(len(couplets_4), 5), dtype=numpy.float64)

    print(
        "\nStep 4: calculating scores for naive couplets from corpus ({0} data points)\n".format(
            len(couplets_4)
        )
    )
    for i, couplet in enumerate(couplets_4):
        # print("evaluating couplet {0}".format(i))
        naive_scores[i] = couplet_scorer.calculate_scores(couplet)

    compute_and_print_stats("naive generator", naive_scores)

    return 0


if __name__ == "__main__":
    sys.exit(main())
