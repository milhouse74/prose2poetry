from .rhyme_score import rhyme_score
import itertools
import pronouncing
import difflib


class CoupletScorer:
    ### combined score weights
    rhyme_weight = 0.5
    stress_weight = 0.5

    @staticmethod
    def calculate_scores(poem_lines):
        ### keep only two-lines poem
        if len(poem_lines) != 2:
            raise ValueError("can only score 2-line poems/couplets")

        ### calculate informations needed for scoring
        num_words = 0
        stress_strings = []
        all_poem_words = []
        last_words = []

        for pl in poem_lines:
            try:
                pwords = pl.split()
            except AttributeError:
                pwords = pl

            num_words += len(pwords)
            last_words.append(pwords[-1])
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
        rhyme_score_ = rhyme_score(last_words[0], last_words[1], penalize_short_word=False)

        ### stress score
        stress_string_score = difflib.SequenceMatcher(
            None, stress_strings[0], stress_strings[1]
        ).ratio()

        ### combined score
        ret = (
            CoupletScorer.rhyme_weight * rhyme_score_
            + CoupletScorer.stress_weight * stress_string_score
        )
        return [
            ret,
            rhyme_score_,
            stress_string_score,
        ]
