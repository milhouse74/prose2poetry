from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import pathlib

def _train_and_save_model(sents, model_path):
    tagged_sents = [TaggedDocument(words=word_tokenize(sent), tags=[str(i)]) for i, sent in enumerate(sents)]
    d2v_model = Doc2Vec(
        tagged_sents, vector_size=128, window=64, min_count=5, sample=1e-2, dm=1, epochs=50
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
            self.d2v_model = _train_and_save_model(
                sents, str(Doc2vecModel.model_path)
            )

    def similarity(self, sent1, sent2):
        return (self.d2v_model.docvecs.similarity_unseen_docs(self.d2v_model, word_tokenize(sent1), word_tokenize(sent2)) + 1) / 2 # normalizing the score between 0-1