#!/usr/bin/env python

import sys
from prose2poetry.corpora import ProseCorpus, GutenbergCouplets, PFCouplets, pairs
from prose2poetry.couplet_score import CoupletScorer
from prose2poetry.generators import NaiveGenerator, MarkovChainGenerator
from prose2poetry.vector_models import FasttextModel
import argparse
import numpy
import random
import multiprocessing
import itertools
from tabulate import tabulate


def compute_stats(scores):
    return (
        numpy.mean(scores, axis=0),
        numpy.std(scores, axis=0),
        numpy.quantile(scores, 0.95, axis=0),
    )


def score_couplets(couplets, scorer):
    scores = numpy.ndarray(shape=(len(couplets), 4), dtype=numpy.float64)

    for i, couplet in enumerate(couplets):
        scores[i] = scorer.calculate_scores(couplet)

    return scores


def main():
    parser = argparse.ArgumentParser(
        prog="prose2poetry",
        description="Evaluate baseline metrics",
    )

    parser.add_argument(
        "--n-eval",
        type=int,
        default=999,
        help="Number of couplets to sample for evaluation from all corpora/generators",
    )

    parser.add_argument(
        "--n-pool",
        type=int,
        default=16,
        help="Size of multiprocessing pool (adjust for number of threads)",
    )

    parser.add_argument(
        "--rand-seed",
        type=int,
        default=42,
        help="Integer seed for rng",
    )

    parser.add_argument(
        "--memory-growth",
        action="store_true",
        help="Allow TF GPU memory growth (useful for nvidia RTX 2xxx cards)",
    )

    args = parser.parse_args()

    pool = multiprocessing.Pool(args.n_pool)

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

    # use random pairs of sentences from our prose corpus for "bad couplets"
    prose_corpus = ProseCorpus()
    couplets_3 = random.sample(list(pairs(prose_corpus.joined_sents)), args.n_eval)

    gen = NaiveGenerator(prose_corpus)
    couplets_4 = gen.generate_couplets(n=args.n_eval)

    ft_model = FasttextModel(prose_corpus)

    seed_words = ['love', 'hate', 'pride', 'prejudice', 'justice', 'romance']

    # get at least 5x top_n semantically similar words to increase the chances of finding good rhyming pairs among them
    semantic_sim_words = ft_model.get_top_n_semantic_similar(
        seed_words, n=(5 * 200)
    )

    # deduplicate
    semantic_sim_words = list(set(semantic_sim_words))

    semantic_sim_words.extend(seed_words)
    rhyme_pairs = itertools.combinations(semantic_sim_words, 2)

    all_results = []
    for rp in rhyme_pairs:
        # use a combined score which incorporates rhyme and semantic score
        all_results.append((ft_model.combined_score(rp[0], rp[1]), rp[0], rp[1]))

    # sort in reverse order
    all_results = sorted(all_results, key=lambda x: x[0], reverse=True)

    gen2 = MarkovChainGenerator(prose_corpus, memory_growth=args.memory_growth)

    # return is a set, cast it to a list
    couplets_5 = list(gen2.generate_couplets(all_results, n=args.n_eval))

    third = int(args.n_eval / 3)

    scores = list(
        pool.starmap(
            score_couplets,
            zip(
                [
                    couplets_1[:third],
                    couplets_1[third : 2 * third],
                    couplets_1[2 * third :],
                    couplets_2[:third],
                    couplets_2[third : 2 * third],
                    couplets_2[2 * third :],
                    couplets_3[:third],
                    couplets_3[third : 2 * third],
                    couplets_3[2 * third :],
                    couplets_4[:third],
                    couplets_4[third : 2 * third],
                    couplets_4[2 * third :],
                    couplets_5[:third],
                    couplets_5[third : 2 * third],
                    couplets_5[2 * third :],
                ],
                itertools.repeat(couplet_scorer),
            ),
        )
    )

    for i in range(0, len(scores), 3):
        scores_ndarray = numpy.concatenate(
            (scores[i], scores[i + 1], scores[i + 2]), axis=0
        )
        stats = compute_stats(scores_ndarray)

        if i == 0:
            print(
                "\nGutenberg, {0} couplets\n----------------------------------------------\n".format(
                    scores_ndarray.shape[0]
                )
            )
        elif i == 3:
            print(
                "\nPoetryFoundation, {0} couplets\n----------------------------------------------\n".format(
                    scores_ndarray.shape[0]
                )
            )
        elif i == 6:
            print(
                "\nProse, {0} couplets\n----------------------------------------------\n".format(
                    scores_ndarray.shape[0]
                )
            )
        elif i == 9:
            print(
                "\nNaive generator, {0} couplets\n----------------------------------------------\n".format(
                    scores_ndarray.shape[0]
                )
            )
        elif i == 12:
            print(
                "\nMarkov generator, {0} couplets\n----------------------------------------------\n".format(
                    scores_ndarray.shape[0]
                )
            )


        headers = [
            "metric",
            "mean",
            "std",
            ".95 quantile",
        ]
        metrics = ["total", "stress", "semantic", "meteor"]
        table = []
        for j, metric_name in enumerate(metrics):
            table.append(
                [
                    metric_name,
                    stats[0][j],
                    stats[1][j],
                    stats[2][j],
                ]
            )

        # get top .95 quantile couplets
        couplet_scores = scores_ndarray[:, 0]
        top95q_indices = numpy.argwhere(couplet_scores >= stats[2][0])
        print(tabulate(table, headers, tablefmt="github"))

        print("\ntop 5% couplets:")
        for idx in top95q_indices:
            if i == 0:
                print(couplets_1[idx[0]])
            elif i == 3:
                print(couplets_2[idx[0]])
            elif i == 6:
                print(couplets_3[idx[0]])
            elif i == 9:
                print(couplets_4[idx[0]])
            elif i == 12:
                print(couplets_5[idx[0]])

    return 0


if __name__ == "__main__":
    sys.exit(main())
