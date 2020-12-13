import itertools
from collections import defaultdict
import pronouncing
import numpy
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, LSTM, GRU, SimpleRNN
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.losses import MeanSquaredError
import tensorflow
from gensim.models import FastText
import math
import pathlib
import matplotlib.pyplot as plt
import markovify
import random
import time
from collections import Counter


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
    model.add(LSTM(64, input_shape=(dataX.shape[1], dataX.shape[2])))
    model.add(Dense(dataY.shape[1], activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="adam")

    model.summary()

    history = model.fit(dataX, dataY, epochs=200, batch_size=128, validation_split=0.20)
    loss_train = history.history["loss"]
    loss_val = history.history["val_loss"]
    # print(numpy.min(loss_val[24:]))

    # show learning graph
    plt.plot(loss_train, label="Training")
    plt.plot(loss_val, label="Validation")
    plt.title("LSTM - Cross-entropy by Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy")
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
        while gen_word_i not in [".", ";", "!", "?"]:
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


class CustomMarkovChainGenerator:
    def __init__(self, prose_corpus):
        self.bigram_suffix_to_trigram = {}
        self.bigram_suffix_to_trigram_weights = {}

        for line in prose_corpus.sents:
            line.insert(0, "<BOS>")
            line.append("<EOS1>")
            line.append("<EOS2>")

            i = len(line) - 1

            while i > 1:
                word3 = line[i]
                word2 = line[i - 1]
                word1 = line[i - 2]
                if (word2, word3) not in self.bigram_suffix_to_trigram:
                    self.bigram_suffix_to_trigram[(word2, word3)] = []
                    self.bigram_suffix_to_trigram_weights[(word2, word3)] = []

                if word1 not in self.bigram_suffix_to_trigram[(word2, word3)]:
                    self.bigram_suffix_to_trigram[(word2, word3)].append(word1)
                    self.bigram_suffix_to_trigram_weights[(word2, word3)].append(1)

                elif word1 in self.bigram_suffix_to_trigram[(word2, word3)]:
                    self.bigram_suffix_to_trigram_weights[(word2, word3)][
                        self.bigram_suffix_to_trigram[(word2, word3)].index(word1)
                    ] += 1
                i = i - 1

    def top_prev_word(self, word2, word3, n=10):
        # this is causing problems
        if (word2, word3) not in self.bigram_suffix_to_trigram:
            words = [word2, word3]
            probs = []
            probs.append(0)
            probs.append(0)
            return words, probs

        curr_bigram_dict = {}
        size = len(self.bigram_suffix_to_trigram[word2, word3])
        i = 0
        denom = 0
        while i < size:
            curr_bigram_dict[
                self.bigram_suffix_to_trigram[word2, word3][i]
            ] = self.bigram_suffix_to_trigram_weights[(word2, word3)][i]
            denom += self.bigram_suffix_to_trigram_weights[(word2, word3)][i]
            i += 1
        k = Counter(curr_bigram_dict)
        most_common = k.most_common(n)
        keys = []
        values = []
        for key, value in most_common:
            keys.append(key)
            values.append(value / denom)
        return keys, values

    def generate_sentences(self, suffix, beam=10):
        suffix = suffix.split()
        word2 = suffix[len(suffix) - 1]
        word3 = suffix[len(suffix) - 2]

        candidates = []
        probs = []
        prev_words, prev_probs = self.top_prev_word(word2, word3, beam)
        top_10 = []
        top_10_probs = []

        i = 0
        while i < len(prev_words):
            temp = []
            for item in suffix:
                temp.append(item)
            temp.append(prev_words[i])
            candidates.append(temp)
            probs.append(prev_probs[i])
            i += 1

        iter = 0
        done = 0
        while len(top_10) < 10:
            # while done == 0:
            big_candidates = []
            big_probs = []

            for t, candidate in enumerate(candidates):
                # for each of the ten current possible sentences
                word2 = candidate[len(candidate) - 1]
                word3 = candidate[len(candidate) - 2]

                prev_words, prev_probs = self.top_prev_word(word2, word3, beam)

                f = 0
                for f, word in enumerate(prev_words):
                    temp = []
                    for item in candidate:
                        temp.append(item)
                    temp.append(word)
                    big_candidates.append(temp)
                    big_probs.append(prev_probs[f] * probs[t])

            # this is currently sorting according to the transitional probabilities alone
            # but to incorporate semantic cohesion it could sort according to some semantic cohesion score combined w/that
            # so above, each time the probability of the candidate sentence is updated
            # the probability I'm currently using could be somehow combined to a semantic similarity score
            # could be overloaded s.t. it compares to either second rhyming word, or the entire first sentence depending on which sentence is being generated
            high = sorted(range(len(big_probs)), key=lambda sub: big_probs[sub])[-beam:]

            candidates = []
            probs = []
            for index in high:

                if big_candidates[index][len(big_candidates[index]) - 1] == "<BOS>":

                    # done = 1
                    return big_candidates[index]
                    # top_10.append(big_candidates[index])
                    # top_10_probs.append(big_probs[index])
                else:
                    candidates.append(big_candidates[index])
                    probs.append(big_probs[index])
                    if iter > 20:
                        return big_candidates[index]
            iter += 1

            # return top_10;
        return top_10

    def generate_couplets(self, rhyming_pairs, n=10):
        ret = set()
        iters = 0
        n_old = 0
        while len(ret) < n:
            for rhyming_pair in rhyming_pairs:
                print("rhyming pair: {0}".format(rhyming_pair))
                # rhyming_pair[0] is the rhyme score
                # tuples of strings are hashable
                sent1 = " ".join(
                    self.generate_sentences("<EOS1> " + rhyming_pair[1])[::-1]
                )

                if "<BOS>" not in sent1:
                    continue

                sent1 = sent1.replace("<EOS1>", "").replace("<BOS>", "")

                sent2 = " ".join(
                    self.generate_sentences("<EOS1> " + rhyming_pair[2])[::-1]
                )

                if "<BOS>" not in sent2:
                    continue
                sent2 = sent2.replace("<EOS1>", "").replace("<BOS>", "")

                ret.add((sent1, sent2))
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
