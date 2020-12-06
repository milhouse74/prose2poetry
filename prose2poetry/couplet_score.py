from .rhyme_score import rhyme_score
import itertools
import pronouncing
import difflib
from nltk.translate.meteor_score import meteor_score
from .vector_models import Doc2vecModel


class CoupletScorer:
    ### combined score weights
    stress_weight = 0.4
    semantic_weight = 0.2
    meteor_weight = 0.2

    def __init__(self, reference_corpus):
        self.semantic_scorer = Doc2vecModel(reference_corpus)
        self.reference_corpus = reference_corpus

    def calculate_scores(self, poem_lines):
        ### keep only two-lines poem
        if len(poem_lines) != 2:
            raise ValueError("can only score 2-line poems/couplets")

        ### calculate informations needed for scoring
        num_words = 0
        stress_strings = []
        all_poem_words = []

        for pl in poem_lines:
            try:
                pwords = pl.split()
            except AttributeError:
                pwords = pl

            num_words += len(pwords)
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

        ### stress score
        stress_string_score = difflib.SequenceMatcher(
            None, stress_strings[0], stress_strings[1]
        ).ratio()

        ### semantic score
        semantic_score = self.semantic_scorer.similarity(poem_lines[0], poem_lines[1])

        ### METEOR score
        meteor_score_1 = meteor_score(self.reference_corpus, poem_lines[0])
        meteor_score_2 = meteor_score(self.reference_corpus, poem_lines[1])

        try:
            meteor_score_combined = (meteor_score_1 + meteor_score_2) / 2.0
        except Exception as e:
            print("failed to get meteor score: {0}".format(str(e)))

        ### combined score
        ret = (
            CoupletScorer.stress_weight * stress_string_score
            + CoupletScorer.semantic_weight * semantic_score
            + CoupletScorer.meteor_weight * meteor_score_combined
        )
        return [
            ret,
            stress_string_score,
            semantic_score,
            meteor_score_combined,
        ]
