import itertools
from collections import defaultdict
import pronouncing
import numpy
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, LSTM, GRU, SimpleRNN
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.losses import MeanSquaredError
import tensorflow
from nltk.corpus import gutenberg
from gensim.models import FastText
import math
import pathlib
import matplotlib.pyplot as plt
import markovify
import random
import time


def _train_and_save_model_lstm_1(prose_corpus, ft_model, vocab, model_path, memory=5):

    data = numpy.asarray(prose_corpus.words)

    # Embedding and one-hot
    data_embedded = []
    data_onehot = []
    for word in numpy.flip(data):  # this is where the backward formatting occurs
        if word in ft_model.wv.vocab:
            data_embedded.append(ft_model.wv.get_vector(word))
            data_onehot.append(vocab.index(word))
    data_embedded = numpy.asarray(data_embedded)
    data_onehot = to_categorical(data_onehot)

    # size of embedding vectors
    n_words = len(vocab)
    n_words_capped = math.floor(n_words / memory) * memory
    n_embedding = len(data_embedded[0])

    # Generate dataset
    dataX = []
    dataY = []
    for i in range(0, n_words_capped - memory):
        dataX.append(data_embedded[i : i + memory])
        dataY.append(data_onehot[i + memory])
    dataX = numpy.asarray(dataX)
    dataY = numpy.asarray(dataY)

    # Create model
    model = Sequential()
    model.add(LSTM(128, input_shape=(dataX.shape[1], dataX.shape[2])))
    model.add(Dense(dataY.shape[1], activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="adam")

    model.summary()

    history = model.fit(dataX, dataY, epochs=200, batch_size=128, validation_split=0.20)
    loss_train = history.history["loss"]
    loss_val = history.history["val_loss"]

    # show learning graph
    plt.plot(loss_train, label="Training")
    plt.plot(loss_val, label="Validation")
    plt.title("Training Curves")
    plt.legend()
    plt.show()

    # Save model
    model.save(model_path)

    return model, loss_train, loss_val


class LSTMModel1:
    models_path = pathlib.Path(__file__).parent.absolute().joinpath("../models")
    model_path = models_path.joinpath("sequence_model_1")

    def __init__(self, prose_corpus, ft_model, memory=5, memory_growth=False):
        # we passed our own FasttextModel wrapper, use the inner ft_model class member
        self.ft_model = ft_model.ft_model

        self.vocab = [key for key in self.ft_model.wv.vocab.keys()]
        self.model = None
        self.memory = memory

        if memory_growth:
            physical_devices = tensorflow.config.list_physical_devices("GPU")
            tensorflow.config.experimental.set_memory_growth(
                physical_devices[0], enable=True
            )

        try:
            self.model = load_model(LSTMModel1.model_path)
        except Exception as e:
            print("could not load model, training...")
            self.model, _, _ = _train_and_save_model_lstm_1(
                prose_corpus,
                self.ft_model,
                self.vocab,
                LSTMModel1.model_path,
                memory=memory,
            )

    def generate_sentence(self, end_word):
        gen_word_i = end_word
        gen_sent = [
            gen_word_i,
        ]
        while gen_word_i not in ['.', ';', '!', '?']:
            memory_adjusted = min(self.memory, len(gen_sent))

            pred = numpy.asarray(
                [self.ft_model.wv[word] for word in gen_sent[-memory_adjusted:]]
            )
            to_predict = numpy.reshape(
                pred,
                (1, memory_adjusted, pred.shape[1]),
            )
            prediction = self.model.predict(to_predict, verbose=0)
            gen_word_i = self.vocab[random.choice(prediction[-1].argsort()[-3:])]
            gen_sent.append(gen_word_i)

        return " ".join(gen_sent[:-1][::-1])

    def generate_couplets(self, rhyming_pairs, n=10):
        ret = set()
        iters = 0
        n_old = 0
        while len(ret) < n:
            print("here we go!")
            for rhyming_pair in rhyming_pairs:
                print("rhyming pair: {0}".format(rhyming_pair))
                # rhyming_pair[0] is the rhyme score
                # tuples of strings are hashable
                ret.add(
                    (
                        self.generate_sentence(rhyming_pair[1]),
                        self.generate_sentence(rhyming_pair[2]),
                    )
                )
                print("currently at {0} couplets".format(len(ret)))

                if len(ret) >= n:
                    return ret
                continue
            # reached end of rhyming pair list
            n_new = len(ret)
            iters += 1
            if n_new == n_old:
                print("exhausted at {0} couplets".format(n_new))
                return ret
            else:
                print("appended {0} new couplets".format(n_new - n_old))
            n_old = n_new

        return ret


class NaiveGenerator:
    def __init__(self, prose_corpus):
        # iterate through the prose corpus
        # find all last words that rhyme, collect these into "couplets"

        self.last_words = defaultdict(list)
        for sent in prose_corpus.sents:
            if not sent:
                continue

            # accumulate all sentences that end with the same last word
            self.last_words[sent[-1]].append(sent)

        last_word_pairs = list(itertools.combinations(self.last_words.keys(), 2))

        self.rhymes = []

        for i, word_pair in enumerate(last_word_pairs):
            if word_pair[1] in pronouncing.rhymes(word_pair[0]):
                self.rhymes.append(word_pair)

    def generate_couplets(self, n=10):
        ret = []
        for pair in self.rhymes[:n]:
            ret.append(
                [
                    # pick the shortest candidates
                    " ".join(min(self.last_words[pair[0]], key=len)),
                    " ".join(min(self.last_words[pair[1]], key=len)),
                ]
            )
        return ret


class MarkovChainGenerator:
    def __init__(self, prose_corpus, memory_growth=False):
        if memory_growth:
            physical_devices = tensorflow.config.list_physical_devices("GPU")
            tensorflow.config.experimental.set_memory_growth(
                physical_devices[0], enable=True
            )

        self.markov_model = markovify.NewlineText(
            "\n".join(prose_corpus.joined_sents_with_punct)
        )

    def generate_sentence(self, end_word):
        return self.markov_model.make_sentence_that_finish(end_word)

    def generate_couplets(self, rhyming_pairs, n=10):
        ret = set()
        iters = 0
        n_old = 0
        while len(ret) < n:
            print("here we go!")
            for rhyming_pair in rhyming_pairs:
                print("rhyming pair: {0}".format(rhyming_pair))
                try:
                    # rhyming_pair[0] is the rhyme score
                    # tuples of strings are hashable
                    ret.add(
                        (
                            self.markov_model.make_sentence_that_finish(
                                rhyming_pair[1]
                            ),
                            self.markov_model.make_sentence_that_finish(
                                rhyming_pair[2]
                            ),
                        )
                    )
                except Exception:
                    pass
                print("currently at {0} couplets".format(len(ret)))

                if len(ret) >= n:
                    return ret
                continue
            # reached end of rhyming pair list
            n_new = len(ret)
            iters += 1
            if n_new == n_old:
                print("exhausted at {0} couplets".format(n_new))
                return ret
            else:
                print("appended {0} new couplets".format(n_new - n_old))
            n_old = n_new

        return ret
