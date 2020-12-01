from gensim.models import FastText
from nltk.corpus import stopwords
from .rhyme_score import rhyme_score
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import pathlib
import pronouncing


### FastText
def _post_filter_processing_ft(list_words, len_word_filter=3):
    return [
        word
        for word in list_words
        if word not in stopwords.words("english")
        and len(word) > len_word_filter
        and word.isalpha()
        and pronouncing.phones_for_word(word)
    ]


def _train_and_save_model_ft(sents, model_path):
    ft_model = FastText(
        sents, size=128, window=32, min_count=5, sample=1e-2, sg=1, iter=50
    )
    ft_model.save(model_path)
    return ft_model


class FasttextModel:
    models_path = pathlib.Path(__file__).parent.absolute().joinpath("../models")
    model_path = models_path.joinpath("fasttext.bin")
    rhyme_weight = 0.80

    def __init__(self, prose_corpus):
        self.ft_model = None
        try:
            self.ft_model = FastText.load(str(FasttextModel.model_path))
        except Exception as e:
            print("could not load model, training...")
            self.ft_model = _train_and_save_model_ft(
                prose_corpus.sents, str(FasttextModel.model_path)
            )

    def get_top_n_semantic_similar(self, seed_words, n=50):
        ret_filtered = []
        for seed_word in seed_words:
            similar_word_tmp = [
                w[0] for w in self.ft_model.wv.most_similar(seed_word, topn=2 * n)
            ]
            ret_filtered.extend(_post_filter_processing_ft(similar_word_tmp)[:n])

        return ret_filtered

    def semantic_score(self, word1, word2):
        return (
            self.ft_model.wv.similarity(w1=word1, w2=word2) + 1
        ) / 2  # normalizing the score between 0-1

    def rhyme_score(self, word1, word2):
        return rhyme_score(word1, word2)

    def combined_score(self, word1, word2, rhyme_weight=None):
        if not rhyme_weight:
            rhyme_weight = FasttextModel.rhyme_weight
        return self.rhyme_score(word1, word2) * rhyme_weight + self.semantic_score(
            word1, word2
        ) * (1 - rhyme_weight)


### Doc2Vec
def _train_and_save_model_d2v(sents, model_path):
    tagged_sents = [
        TaggedDocument(words=word_tokenize(sent), tags=[str(i)])
        for i, sent in enumerate(sents)
    ]
    d2v_model = Doc2Vec(
        tagged_sents,
        vector_size=128,
        window=64,
        min_count=5,
        sample=1e-2,
        dm=1,
        epochs=50,
    )
    d2v_model.save(model_path)
    return d2v_model


class Doc2vecModel:
    models_path = pathlib.Path(__file__).parent.absolute().joinpath("../models")
    model_path = models_path.joinpath("doc2vec.bin")

    def __init__(self, sents):
        self.d2v_model = None
        try:
            self.d2v_model = Doc2Vec.load(str(Doc2vecModel.model_path))
        except Exception as e:
            print("could not load model, training...")
            self.d2v_model = _train_and_save_model_d2v(
                sents, str(Doc2vecModel.model_path)
            )

    def similarity(self, sent1, sent2):
        return (
            self.d2v_model.docvecs.similarity_unseen_docs(
                self.d2v_model, word_tokenize(sent1), word_tokenize(sent2)
            )
            + 1
        ) / 2  # normalizing the score between 0-1
