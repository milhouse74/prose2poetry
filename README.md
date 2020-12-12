# prose2poetry

COMP 550 NLP Fall 2020 final project: _Prose2poetry - Generating poetry from prose_

## Licenses

The code is licensed under the MIT license. Supplemental data (baseline corpora, etc.) have their own attributions in [data/](./data/README.md). Models are trained from scratch and stored in `./models`. The models dir using a single novel as an input corpus takes a combined 2GB of space, so these are not included in the repo. Retraining the models from scratch doesn't take too much time (~10 minutes).

## Install dependencies

Most of the dependencies are in the `requirements.txt` file. Install in a virtualenv (or your tool of choosing).

```
$ pip install -r ./requirements.txt
```

## Code structure

There are 3 runnable scripts:
* `evaluate_metrics.py` - load and score baselines and generated couplets
* `evaluate_doc2vec.py` - evaluate semantic similarity in the generated couplets
* `prose2poetry.py` - take seed words as an input and produce an output poem

The important code is in the embedded [prose2poetry](./prose2poetry) library:
* `generators.py` for poetry generators including an LSTM and Markov chain model
* `rhyme_score.py` contains our custom `rhyme_score` function using phoneme data from the CMUdict
* `couplet_score.py` contains the couplet scorer which incorporates rhyme score on the end words, and a syllabic meter score
* `vector_models.py` contains gensim Fasttext and doc2vec embedding models + training and loading code
* `corpora.py` contains some classes to faciliate the loading and filtering of couplets from nltk's Gutenberg corpus, the [Gutenberg Poetry corpus](https://github.com/aparrish/gutenberg-poetry-corpus), and [PoetryFoundation](https://www.kaggle.com/tgdivy/poetry-foundation-poems) corpus (included in [data](./data))

## Usage

When initially cloning the project, the data dir contains the baseline corpora stored with Git-LFS. You should confirm that the size of the data directory is 75M. If it isn't, you may need to run `git-lfs pull`.

The models directory on a fresh clone is empty. This is where FastText, doc2vec, and the LSTM models are stored after training. The first time you run `./evaluate_metrics.py` or `./prose2poetry.py`, the models will be trained and saved. To reset the training (e.g. if changing the input corpus), delete the contents of the models directory.

### Example

Generating couplets from the novel _Emma_ by Jane Austen, using the seed word "love":

```
$ ./prose2poetry.py love
Markov chain generated couplet:
        ('He seems every thing -- Who is in great spirits one morning to enjoy .-- She saw that Enscombe could not often so ill - tempered men that ever was , therefore , to the end of it ; and on the occasion of much present enjoyment to be the means of promoting it , I have heard every thing into sad uncertainty', 'Respect for right conduct is felt by every body for their not meaning to make more than common certainty')
        score: 0.7116666666666667
LSTM generated couplet:
        (', to for , and and to , from the with the his his s his nervous uncertainty', 'and it a - certainty')
        score: 0.685
```
