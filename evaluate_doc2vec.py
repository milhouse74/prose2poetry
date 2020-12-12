#!/usr/bin/env python

import sys
from prose2poetry.corpora import ProseCorpus
from prose2poetry.generators import MarkovChainGenerator, LSTMModel1
from prose2poetry.vector_models import FasttextModel, Doc2vecModel
import argparse
import numpy
import random
import multiprocessing
import itertools
from tabulate import tabulate
import pronouncing


def compute_stats(scores):
    return (
        numpy.mean(scores, axis=0),
        numpy.std(scores, axis=0),
        numpy.quantile(scores, 0.95, axis=0),
    )


def score_couplets(couplets, doc2vec):
    scores = numpy.ndarray(shape=(len(couplets), 1), dtype=numpy.float64)

    for i, couplet in enumerate(couplets):
        scores[i] = doc2vec.similarity(couplet[0], couplet[1])

    return scores


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
        "--ft-model",
        type=int,
        default=1,
        help="Use FastText model to create word pairs",
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

    # use random pairs of sentences from our prose corpus for "bad couplets"
    prose_corpus = ProseCorpus()

    ft_model = FasttextModel(prose_corpus)
    d2v_model = Doc2vecModel(prose_corpus)

    seed_words = ["love", "hate", "pride", "prejudice", "justice", "romance"]

    # Use FastText semantic or not
    if args.ft_model == 1:
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
    else:
        # get list of words that rhymes with the seed words
        all_results = []
        for seed_word in seed_words:
            for word in pronouncing.rhymes(seed_word):
                all_results.append((1, seed_word, word))

    gen2 = MarkovChainGenerator(prose_corpus, memory_growth=args.memory_growth)
    gen3 = LSTMModel1(prose_corpus, ft_model, memory_growth=args.memory_growth)

    # return is a set, cast it to a list
    couplets_5 = list(gen2.generate_couplets(all_results, n=args.n_eval))

    couplets_6 = list(gen3.generate_couplets(all_results, n=args.n_eval))

    quarter = int(args.n_eval / 4)

    scores = list(
        pool.starmap(
            score_couplets,
            zip(
                [
                    couplets_5[:quarter],
                    couplets_5[quarter : 2 * quarter],
                    couplets_5[2 * quarter : 3 * quarter],
                    couplets_5[3 * quarter :],
                    couplets_6[:quarter],
                    couplets_6[quarter : 2 * quarter],
                    couplets_6[2 * quarter : 3 * quarter],
                    couplets_6[3 * quarter :],
                ],
                itertools.repeat(d2v_model),
            ),
        )
    )

    for i in range(0, len(scores), 4):
        scores_ndarray = numpy.concatenate(
            (scores[i], scores[i + 1], scores[i + 2], scores[i + 3]), axis=0
        )
        stats = compute_stats(scores_ndarray)

        if i == 0:
            print(
                "\nMarkov generator, {0} couplets\n----------------------------------------------\n".format(
                    scores_ndarray.shape[0]
                )
            )
        elif i == 4:
            print(
                "\nLSTM generator, {0} couplets\n----------------------------------------------\n".format(
                    scores_ndarray.shape[0]
                )
            )

        headers = [
            "metric",
            "mean",
            "std",
            ".95 quantile",
        ]
        metrics = ["doc2vec"]
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

        print(tabulate(table, headers, tablefmt="github"))

    return 0


if __name__ == "__main__":
    sys.exit(main())
