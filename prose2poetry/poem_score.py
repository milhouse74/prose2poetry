from .rhyme_score import rhyme_score
import itertools
import pronouncing
import difflib
import torch
import nltk
from nlgeval import NLGEval
from .semantic_similarity import SemanticSimilarity


class PoemScorer:
    # rhyme score weight
    rhyme_weight = 1.5
    stress_weight = 0.5
    semantic_weight = 0.0  # 0.25
    meteor_weight = 0.25

    def __init__(self, prose_corpus):
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

        self.semantic_scorer = SemanticSimilarity(prose_corpus)
        self.prose_corpus = prose_corpus

    def score_poem(self, poem_lines):
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

        last_word_combinations = itertools.combinations(last_words, 2)

        last_word_rhyme_score = 0.0
        for l in last_word_combinations:
            last_word_rhyme_score += rhyme_score(l[0], l[1])

        stress_string_combinations = itertools.combinations(stress_strings, 2)

        stress_string_score = 0.0
        for s in stress_string_combinations:
            stress_string_score += difflib.SequenceMatcher(None, s[0], s[1]).ratio()

        semantic_score = 0.0

        tmp = self.semantic_scorer.similarity(poem_lines[0], poem_lines[1])
        semantic_score += tmp

        nlg_scores_1 = self.nlgeval.compute_individual_metrics(
            self.prose_corpus.joined_sents, poem_lines[0]
        )
        nlg_scores_2 = self.nlgeval.compute_individual_metrics(
            self.prose_corpus.joined_sents, poem_lines[1]
        )

        meteor_score = 0.0
        try:
            meteor_score = nlg_scores_1["METEOR"] + nlg_scores_2["METEOR"]
        except Exception as e:
            print("failed to get meteor score: {0}".format(str(e)))

        # normalize by poem length in words
        ret = (
            PoemScorer.rhyme_weight * last_word_rhyme_score
            + PoemScorer.stress_weight * stress_string_score
            + PoemScorer.semantic_weight * semantic_score
            + PoemScorer.meteor_weight * meteor_score
        ) / num_words

        return ret
