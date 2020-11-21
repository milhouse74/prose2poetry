from gensim.models import FastText
from .rhyme_score import rhyme_score
import pathlib


def _train_and_save_model(sents, model_path):
    ft_model = FastText(
        sents, size=100, window=40, min_count=5, sample=1e-2, sg=1, iter=100
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
            self.ft_model = _train_and_save_model(
                prose_corpus.sents, str(FasttextModel.model_path)
            )

    def get_top_n_semantic_similar(self, seed_words, n=50):
        ret = []
        for seed_word in seed_words:
            ret.extend([w[0] for w in self.ft_model.wv.most_similar(seed_word, topn=n)])
        return ret

    def semantic_score(self, word1, word2):
        return self.ft_model.wv.similarity(w1=word1, w2=word2)

    def rhyme_score(self, word1, word2):
        return rhyme_score(word1, word2)

    def combined_score(self, word1, word2, rhyme_weight=None):
        if not rhyme_weight:
            rhyme_weight = FasttextModel.rhyme_weight
        return (
            self.rhyme_score(word1, word2) * rhyme_weight
            + self.semantic_score(word1, word2) * (1 - rhyme_weight) / 2
        )
