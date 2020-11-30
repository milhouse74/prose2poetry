from .rhyme_score import rhyme_score
import itertools
import pronouncing
import difflib
import torch
import nltk
from nlgeval import NLGEval
from .doc2vec_model import Doc2vecModel


class CoupletScorer:
    rhyme_weight = 0.7
    stress_weight = 0.1
    semantic_weight = 0.1
    meteor_weight = 0.1

    def __init__(self, reference_corpus):
        # use METEOR, it's the best-recommended by the paper
        self.nlgeval = NLGEval(
            no_glove=True,
            no_skipthoughts=True,
            metrics_to_omit={
                "CIDEr",
                "ROUGE_L",
                "Bleu_1",
                "Bleu_2",
                "Bleu_3",
                "Bleu_4",
            },
        )

        self.semantic_scorer = Doc2vecModel(reference_corpus)
        self.reference_corpus = reference_corpus

    def calculate_scores(self, poem_lines):
        if len(poem_lines) != 2:
            raise ValueError("can only score 2-line poems/couplets")

        num_words = 0
        last_words = set()
        stress_strings = []
        all_poem_words = []

        for pl in poem_lines:
            try:
                pwords = pl.split()
            except AttributeError:
                pwords = pl

            num_words += len(pwords)
            last_words.add(pwords[-1])

            all_poem_words.extend(pwords)

            stress_string = ""
            for pword in pwords:
                try:
                    stress_string += pronouncing.stresses(
                        pronouncing.phones_for_word(pword)[0]
                    )
                except:
                    pass

            stress_strings.append(stress_string)

        ### rhyme score
        last_word_combinations = itertools.combinations(last_words, 2)

        last_word_rhyme_score = 0.0
        for l in last_word_combinations:
            last_word_rhyme_score += rhyme_score(l[0], l[1])

        ### stress score
        stress_string_combinations = list(itertools.combinations(stress_strings, 2))

        stress_string_score = 0.0
        for s in stress_string_combinations:
            stress_string_score += difflib.SequenceMatcher(None, s[0], s[1]).ratio()

        stress_string_score /= len(stress_string_combinations)

        ### semantic score
        semantic_score = self.semantic_scorer.similarity(poem_lines[0], poem_lines[1])

        ### METEOR score
        nlg_scores_1 = self.nlgeval.compute_individual_metrics(
            self.reference_corpus, poem_lines[0]
        )
        nlg_scores_2 = self.nlgeval.compute_individual_metrics(
            self.reference_corpus, poem_lines[1]
        )

        meteor_score = 0.0
        try:
            meteor_score = (nlg_scores_1["METEOR"] + nlg_scores_2["METEOR"]) / 2.0
        except Exception as e:
            print("failed to get meteor score: {0}".format(str(e)))

        ### combined score
        ret = (
            CoupletScorer.rhyme_weight * last_word_rhyme_score
            + CoupletScorer.stress_weight * stress_string_score
            + CoupletScorer.semantic_weight * semantic_score
            + CoupletScorer.meteor_weight * meteor_score
        )

        return [ret, last_word_rhyme_score, stress_string_score, semantic_score, meteor_score]
