# prose2poetry

COMP 550 NLP Fall 2020 final project: _Prose2poetry - Generating poetry from prose_

## Licenses

The code is licensed under the MIT license. Supplemental data (baseline corpora, etc.) have their own attributions in [data/](./data/README.md). Models are trained from scratch (to not include ~1+GB files in the repo) and stored in `./models`.

## Install dependencies

Most of the dependencies are in the `requirements.txt` file. Install in a virtualenv (or your tool of choosing). There is also a custom step in `extra_setup.sh` for installing and configuring the [Maluuba/nlg-eval](https://github.com/Maluuba/nlg-eval) library.

```
$ pip install -r ./requirements.txt
$ ./extra_setup.sh # to install nlg-eval
```

## Code structure

There are 2 runnable scripts:
* `evaluate_metrics.py` - load and score baselines and generated couplets
* `prose2poetry.py` - take seed words as an input and produce an output poem

The important code is in the embedded [prose2poetry](./prose2poetry) library:
* `generators.py` for poetry generators
* `rhyme_score.py` contains our custom `rhyme_score` function, which incorporates phonemic substitution from [Phonemic Similarity Metrics to Compare Pronunciation Methods](https://homes.cs.washington.edu/~bhixon/papers/phonemic_similarity_metrics_Interspeech_2011.pdf) into the score
* `fasttext_model.py` for the genism Fasttext word pair generation, incorporating a semantic score and the above rhyme_score in a weighted combination
* `semantic_score.py` contains an implementation of [Sentence Similarity Based on Semantic Nets and Corpus Statistics](http://ants.iis.sinica.edu.tw/3BkMJ9lTeWXTSrrvNoKNFDxRm3zFwRR/55/Sentence%20Similarity%20Based%20on%20Semantic%20Nets%20and%20corpus%20statistics.pdf) (borrowed from [this implementation](https://github.com/chanddu/Sentence-similarity-based-on-Semantic-nets-and-Corpus-Statistics-))
* `couplet_score.py` contains the couplet scorer, which incorporates METEOR (from Maluuba/nlg-eval), semantic score (above), rhyme score (on the last words), and the edit distance from the string of stresses (produced by [pronouncingpy](https://github.com/aparrish/pronouncingpy))
* `corpora.py` contains some classes to faciliate the loading and filtering of couplets from nltk's Gutenberg corpus, the [Gutenberg Poetry corpus](https://github.com/aparrish/gutenberg-poetry-corpus), and [PoetryFoundation](https://www.kaggle.com/tgdivy/poetry-foundation-poems) corpus (included in [data](./data))

### Generate pairs of rhyming words

Generate pairs of rhyming words from a prose corpus (default: Jane Austen - Emma, available in nltk's Gutenberg corpus):

```
from prose2poetry.corpora import ProseCorpus
from prose2poetry.fasttext_model import FasttextModel
import itertools

# use default prose corpus - gutenberg novels from Jane Austen
corpus = ProseCorpus()

# use prose corpus as input to fasttextmodel
ft_model = FasttextModel(corpus)

seed_words = ['pride', 'prejudice']

# get top n semantically similar words from the fasttext model trained on the prose corpus
semantic_sim_words = ft_model.get_top_n_semantic_similar(
    seed_words, n=50
)

# deduplicate
semantic_sim_words = list(set(semantic_sim_words))

# add seed words into the list
semantic_sim_words.extend(seed_words)
pairs = itertools.combinations(semantic_sim_words, 2)

all_results = []
for p in pairs:
    # use a combined score which incorporates rhyme and semantic score
    all_results.append((ft_model.combined_score(p[0], p[1]), p[0], p[1]))

# sort in reverse order
all_results = sorted(all_results, key=lambda x: x[0], reverse=True)

# print top 10 results by combined score of semantic similarity and rhyme
print("top 10 results for seed words {0}".format(seed_words))
    for a in all_results[: 10]:
        print(
            "combined (semantic, rhyme) score of {0}, {1}: {2}".format(a[1], a[2], a[0])
        )
```

Output:
```
top 10 results for seed words ['pride', 'prejudice']
combined (semantic, rhyme) score of farmer, warmer: 0.8256660335040193
combined (semantic, rhyme) score of bride, pride: 0.7219737315173498
combined (semantic, rhyme) score of coolly, really: 0.6358651270910091
combined (semantic, rhyme) score of notice, service: 0.6341798802223162
combined (semantic, rhyme) score of coolly, wholly: 0.6287796542687313
combined (semantic, rhyme) score of respectable, ridiculous: 0.6180500689991171
combined (semantic, rhyme) score of ridiculous, affectionate: 0.6167437171579526
combined (semantic, rhyme) score of respectable, affectionate: 0.6124035327875881
combined (semantic, rhyme) score of wholly, really: 0.6097792640191385
combined (semantic, rhyme) score of what, but: 0.598802176363053
```

### Couplet scoring

The `CoupletScorer` class requires a corpus as an input. In this project we use the filtered Gutenberg couplets (5000 randomly selected) as the input. These are used as the base corpora of the semantic similarity and METEOR NLG evaluation metrics.

This is a tricky problem since any NLG metrics (which we need by definition since we are generating couplets) require a base corpus against which to compare and generate metrics.
