#!/usr/bin/env python

import sys
import itertools
from prose2poetry.fasttext_model import FasttextModel
from prose2poetry.couplet_score import CoupletScorer
from prose2poetry.corpora import ProseCorpus
import argparse


def main():
    parser = argparse.ArgumentParser(
        prog="prose2poetry",
        description="Generate rhyming couplets from prose corpora",
    )

    parser.add_argument(
        "--include-seed-word",
        action="store_true",
        help="Use seed word in poem, otherwise use any permutation of words from the same universe (semantically)",
    )

    parser.add_argument("seed_words", nargs="+", help="seed word")
    args = parser.parse_args()

    # use default prose corpus - gutenberg novels from Jane Austen
    corpus = ProseCorpus()

    # use prose corpus as input to various internal classes
    ft_model = FasttextModel(corpus)
    couplet_scorer = CoupletScorer(corpus)

    semantic_sim_words = ft_model.get_top_n_semantic_similar(args.seed_words)

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

    print("top 10 results for seed words {0}".format(args.seed_words))
    for a in all_results[: args.top_n]:
        print(
            "combined (semantic, rhyme) score of {0}, {1}: {2}".format(a[1], a[2], a[0])
        )

    poems = [
        ["no sense makes this", "ignorance chicken Buckingham bliss"],
        [
            "would that i could see for one last time",
            "your lovely face in shadows sublime",
        ],
        [
            "if only i could see for one last time",
            "your lovely face in shadows sublime",
        ],
        ["interior crocodile alligator", "i drive a Chevrolet movie theater"],
        [
            "in the brightest core of the burning flame",
            "his flesh melted - he was never the same",
        ],
        ["this doesnt even rhyme", "its a waste of effort"],
        ["this does rhyme", "a good use of our time"],
    ]

    for p in poems:
        print("evaluating poem:\n{0}\nscore: {1}".format(p, couplet_scorer(p)))

    return 0


if __name__ == "__main__":
    sys.exit(main())
