#!/usr/bin/env python

import sys
import itertools
from prose2poetry.fasttext_model import FasttextModel
from prose2poetry.prose_corpus import ProseCorpus
import argparse


def main():
    parser = argparse.ArgumentParser(
        prog="prose2poetry",
        description="Generate rhyming couplets from prose corpora",
    )

    parser.add_argument(
        "--include-seed-word", action="store_true", help="Use seed word in poem, otherwise use any permutation of words from the same universe (semantically)"
    )

    parser.add_argument(
        "--top-n", type=int, default=10, help="Only consider the top n scoring pairs"
    )

    parser.add_argument("seed_words", nargs='+', help="seed word")
    args = parser.parse_args()

    # use default prose corpus - gutenberg novels from Jane Austen
    corpus = ProseCorpus()

    ft_model = FasttextModel(corpus)

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

    print('top 10 results for seed words {0}'.format(args.seed_words))
    for a in all_results[:args.top_n]:
        print("combined (semantic, rhyme) score of {0}, {1}: {2}".format(a[1], a[2], a[0]))

    return 0


if __name__ == "__main__":
    sys.exit(main())
