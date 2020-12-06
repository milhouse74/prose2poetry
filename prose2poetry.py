#!/usr/bin/env python

import sys
import itertools
from prose2poetry.vector_models import FasttextModel
from prose2poetry.couplet_score import CoupletScorer
from prose2poetry.corpora import ProseCorpus, GutenbergCouplets
from prose2poetry.generators import MarkovChainGenerator
import argparse
import random


def main():
    parser = argparse.ArgumentParser(
        prog="prose2poetry",
        description="Generate rhyming couplets from prose corpora",
    )

    parser.add_argument(
        "--top-n",
        type=int,
        default=200,
        help="Top n combined rhyme/semantic scores to return",
    )

    parser.add_argument(
        "--include-seed-word",
        action="store_true",
        help="Use seed word in poem, otherwise use any permutation of words from the same universe (semantically)",
    )

    parser.add_argument(
        "--memory-growth",
        action="store_true",
        help="Allow TF GPU memory growth (useful for nvidia RTX 2xxx cards)",
    )

    parser.add_argument("seed_words", nargs="+", help="seed word")

    parser.add_argument(
        "--rand-seed",
        type=int,
        default=42,
        help="Integer seed for rng",
    )

    args = parser.parse_args()

    # set up random seed to replicate
    random.seed(args.rand_seed)

    # use default prose corpus - gutenberg novels from Jane Austen
    corpus = ProseCorpus()

    # use prose corpus as input to various internal classes
    ft_model = FasttextModel(corpus)

    couplet_gold_standard = GutenbergCouplets()

    # evaluate couplets against the gold standard of gutenberg couplets
    couplet_scorer = CoupletScorer(
        couplet_gold_standard.couplets_flat_list(n_random_couplets=5000)
    )

    # get at least 5x top_n semantically similar words to increase the chances of finding good rhyming pairs among them
    semantic_sim_words = ft_model.get_top_n_semantic_similar(
        args.seed_words, n=5 * args.top_n
    )

    # deduplicate
    semantic_sim_words = list(set(semantic_sim_words))

    pairs = None
    if not args.include_seed_word:
        # add seed word but don't force it to be included
        semantic_sim_words.extend(args.seed_words)
        pairs = itertools.combinations(semantic_sim_words, 2)
    else:
        # force the seed word to be in the poem
        pairs = []
        for seed_word in args.seed_words:
            pairs.extend([(seed_word, x) for x in semantic_sim_words if x != seed_word])

    all_results = []
    for p in pairs:
        # use a combined score which incorporates rhyme and semantic score
        all_results.append((ft_model.combined_score(p[0], p[1]), p[0], p[1]))

    # sort in reverse order
    all_results = sorted(all_results, key=lambda x: x[0], reverse=True)

    generator = MarkovChainGenerator(corpus, memory_growth=args.memory_growth)

    couplets = generator.generate_couplets(all_results, n=1000)
    for couplet in couplets:
        print("Markov chain generated couplet:\n\t{0}".format(couplet))
        print("\tscore: {0}".format(couplet_scorer.calculate_scores(couplet)[0]))

    return 0


if __name__ == "__main__":
    sys.exit(main())
