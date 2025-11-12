# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2025 Argmax, Inc. All Rights Reserved.

import unittest

from argmaxtools.utils import get_logger

from openbench.metric import WordErrorRate
from openbench.pipeline_prediction import Transcript


logger = get_logger(__name__)


def create_transcript(text: str, speakers: str | None = None) -> Transcript:
    """Create a Transcript from text and optional speaker strings."""
    if speakers is None:
        return Transcript.from_words_info(
            words=text.split(),
            start=None,
            end=None,
            speaker=None,
        )
    else:
        return Transcript.from_words_info(
            words=text.split(),
            start=None,
            end=None,
            speaker=speakers.split(),
        )


class TestWER(unittest.TestCase):
    def setUp(self) -> None:
        self.wer = WordErrorRate(use_text_normalizer=False)

    def tearDown(self) -> None:
        self.wer = None

    def _compute_wer(
        self,
        reference_text: str,
        hypothesis_text: str,
    ) -> float:
        reference = create_transcript(reference_text)
        hypothesis = create_transcript(hypothesis_text)

        result = self.wer(reference=reference, hypothesis=hypothesis)
        return result

    def test_wer_perfect_match(self) -> None:
        """Test WER with perfect transcription."""
        reference_text = "hello world good morning"
        hypothesis_text = "hello world good morning"

        result = self._compute_wer(
            reference_text=reference_text,
            hypothesis_text=hypothesis_text,
        )
        self.assertEqual(result, 0.0, "Perfect match should have WER of 0.0")

    def test_wer_single_substitution(self) -> None:
        """Test WER with a single word substitution."""
        reference_text = "hello world good morning"
        hypothesis_text = "hello world bad morning"

        result = self._compute_wer(
            reference_text=reference_text,
            hypothesis_text=hypothesis_text,
        )
        self.assertAlmostEqual(
            result,
            0.25,
            places=4,
            msg="WER should be 0.25 (1 substitution / 4 words)",
        )

    def test_wer_single_deletion(self) -> None:
        """Test WER with a single word deletion."""
        reference_text = "hello world good morning"
        hypothesis_text = "hello world morning"

        result = self._compute_wer(
            reference_text=reference_text,
            hypothesis_text=hypothesis_text,
        )
        self.assertAlmostEqual(
            result,
            0.25,
            places=4,
            msg="WER should be 0.25 (1 deletion / 4 words)",
        )

    def test_wer_single_insertion(self) -> None:
        """Test WER with a single word insertion."""
        reference_text = "hello world good morning"
        hypothesis_text = "hello world good morning friend"

        result = self._compute_wer(
            reference_text=reference_text,
            hypothesis_text=hypothesis_text,
        )
        self.assertAlmostEqual(
            result,
            0.25,
            places=4,
            msg="WER should be 0.25 (1 insertion / 4 words)",
        )

    def test_wer_multiple_substitutions(self) -> None:
        """Test WER with multiple word substitutions."""
        reference_text = "the quick brown fox"
        hypothesis_text = "the fast red fox"

        result = self._compute_wer(
            reference_text=reference_text,
            hypothesis_text=hypothesis_text,
        )
        self.assertAlmostEqual(
            result,
            0.5,
            places=4,
            msg="WER should be 0.5 (2 substitutions / 4 words)",
        )

    def test_wer_multiple_deletions(self) -> None:
        """Test WER with multiple word deletions."""
        reference_text = "hello world good morning friend"
        hypothesis_text = "hello morning"

        result = self._compute_wer(
            reference_text=reference_text,
            hypothesis_text=hypothesis_text,
        )
        self.assertAlmostEqual(
            result,
            0.6,
            places=4,
            msg="WER should be 0.6 (3 deletions / 5 words)",
        )

    def test_wer_multiple_insertions(self) -> None:
        """Test WER with multiple word insertions."""
        reference_text = "hello world"
        hypothesis_text = "hello beautiful amazing world today"

        result = self._compute_wer(
            reference_text=reference_text,
            hypothesis_text=hypothesis_text,
        )
        self.assertAlmostEqual(
            result,
            1.5,
            places=4,
            msg="WER should be 1.5 (3 insertions / 2 words)",
        )

    def test_wer_completely_wrong_text(self) -> None:
        """Test WER with completely incorrect transcription."""
        reference_text = "hello world good morning"
        hypothesis_text = "foo bar baz qux"

        result = self._compute_wer(
            reference_text=reference_text,
            hypothesis_text=hypothesis_text,
        )
        self.assertEqual(result, 1.0, "WER should be 1.0 when all words are substituted")

    def test_wer_empty_hypothesis(self) -> None:
        """Test WER with empty hypothesis."""
        reference_text = "hello world good morning"
        hypothesis_text = ""

        result = self._compute_wer(
            reference_text=reference_text,
            hypothesis_text=hypothesis_text,
        )
        self.assertEqual(result, 1.0, "WER should be 1.0 when hypothesis is empty (all deletions)")

    def test_wer_empty_reference(self) -> None:
        """Test WER with empty reference."""
        reference_text = ""
        hypothesis_text = "hello world"

        # jiwer raises ValueError when reference is empty
        reference = create_transcript(reference_text)
        hypothesis = create_transcript(hypothesis_text)

        with self.assertRaises(ValueError):
            self.wer(reference=reference, hypothesis=hypothesis)

    def test_wer_empty_both(self) -> None:
        """Test WER with both reference and hypothesis empty."""
        reference_text = ""
        hypothesis_text = ""

        # jiwer raises ValueError when reference is empty
        reference = create_transcript(reference_text)
        hypothesis = create_transcript(hypothesis_text)

        with self.assertRaises(ValueError):
            self.wer(reference=reference, hypothesis=hypothesis)

    def test_wer_mixed_errors(self) -> None:
        """Test WER with mixed substitution, insertion, and deletion errors."""
        reference_text = "the quick brown fox jumps over the lazy dog"
        hypothesis_text = "the fast brown fox jumping over lazy dog end"

        result = self._compute_wer(
            reference_text=reference_text,
            hypothesis_text=hypothesis_text,
        )
        # Expected: 2 substitutions (quick->fast, jumps->jumping),
        #           1 deletion (the), 1 insertion (end)
        # Total errors: 4, Total words: 9
        # WER = 4/9 ≈ 0.444
        self.assertAlmostEqual(
            result,
            4 / 9,
            places=3,
            msg="WER should be ~0.444 with mixed errors",
        )

    def test_wer_case_sensitivity(self) -> None:
        """Test WER with different cases (without normalizer)."""
        reference_text = "Hello World"
        hypothesis_text = "hello world"

        result = self._compute_wer(
            reference_text=reference_text,
            hypothesis_text=hypothesis_text,
        )
        # Without normalizer, case matters, so we get 2 substitutions
        self.assertEqual(result, 1.0, "WER should be 1.0 when case differs without normalizer")

    def test_wer_with_punctuation(self) -> None:
        """Test WER with punctuation (without normalizer)."""
        reference_text = "hello world"
        hypothesis_text = "hello, world!"

        result = self._compute_wer(
            reference_text=reference_text,
            hypothesis_text=hypothesis_text,
        )
        # Without normalizer, punctuation is treated as part of the word
        self.assertEqual(result, 1.0, "WER should be 1.0 when punctuation differs without normalizer")

    def test_wer_with_normalizer(self) -> None:
        """Test WER with text normalization enabled."""
        wer_normalized = WordErrorRate(use_text_normalizer=True)

        reference_text = "Hello, World! It's a beautiful day."
        hypothesis_text = "hello world its a beautiful day"

        reference = create_transcript(reference_text)
        hypothesis = create_transcript(hypothesis_text)

        result = wer_normalized(reference=reference, hypothesis=hypothesis)

        # With normalizer, punctuation and case should be handled
        self.assertLess(result, 0.3, "WER should be low with normalizer handling punctuation and case")

    def test_wer_with_contractions(self) -> None:
        """Test WER with contractions using normalizer."""
        wer_normalized = WordErrorRate(use_text_normalizer=True)

        reference_text = "I can't believe it's not butter"
        hypothesis_text = "I cannot believe it is not butter"

        reference = create_transcript(reference_text)
        hypothesis = create_transcript(hypothesis_text)

        result = wer_normalized(reference=reference, hypothesis=hypothesis)

        # With normalizer, contractions are expanded, so there should be some error reduction
        # but not perfect match due to word count differences
        self.assertLess(result, 0.5, "WER should be reduced with normalizer handling contractions")

    def test_wer_with_numbers(self) -> None:
        """Test WER with numbers."""
        reference_text = "I have 3 apples and 5 oranges"
        hypothesis_text = "I have three apples and five oranges"

        result = self._compute_wer(
            reference_text=reference_text,
            hypothesis_text=hypothesis_text,
        )
        # Without normalizer, numbers vs words are different: 2 substitutions out of 7 words
        self.assertAlmostEqual(
            result,
            2 / 7,
            places=3,
            msg="WER should account for number-word differences (2/7)",
        )

    def test_wer_order_matters(self) -> None:
        """Test that WER considers word order."""
        reference_text = "the cat sat on the mat"
        hypothesis_text = "the mat on sat cat the"

        result = self._compute_wer(
            reference_text=reference_text,
            hypothesis_text=hypothesis_text,
        )
        # Words are out of order, should have non-zero WER
        self.assertGreater(result, 0.0, "WER should be > 0 when word order is wrong")

    def test_wer_repeated_words(self) -> None:
        """Test WER with repeated words."""
        reference_text = "hello hello world"
        hypothesis_text = "hello world world"

        result = self._compute_wer(
            reference_text=reference_text,
            hypothesis_text=hypothesis_text,
        )
        # Optimal alignment: 1 substitution (hello->world at position 2) = 1 error / 3 words
        self.assertAlmostEqual(
            result,
            1 / 3,
            places=3,
            msg="WER should handle repeated words correctly",
        )

    def test_wer_single_word(self) -> None:
        """Test WER with single word reference and hypothesis."""
        reference_text = "hello"
        hypothesis_text = "hello"

        result = self._compute_wer(
            reference_text=reference_text,
            hypothesis_text=hypothesis_text,
        )
        self.assertEqual(result, 0.0, "WER should be 0.0 for single matching word")

    def test_wer_single_word_different(self) -> None:
        """Test WER with single word reference and different hypothesis."""
        reference_text = "hello"
        hypothesis_text = "goodbye"

        result = self._compute_wer(
            reference_text=reference_text,
            hypothesis_text=hypothesis_text,
        )
        self.assertEqual(result, 1.0, "WER should be 1.0 for single different word")

    def test_wer_long_text(self) -> None:
        """Test WER with longer text passages."""
        reference_text = "the quick brown fox jumps over the lazy dog while the cat watches from a nearby tree"
        hypothesis_text = "the quick brown fox leaps over the lazy dog while the cat watches from a distant tree"

        result = self._compute_wer(
            reference_text=reference_text,
            hypothesis_text=hypothesis_text,
        )
        # 2 substitutions: jumps->leaps, nearby->distant
        # Total words: 17
        # WER = 2/17 ≈ 0.1176
        self.assertAlmostEqual(
            result,
            2 / 17,
            places=3,
            msg="WER should be ~0.1176 with 2 substitutions in 17 words",
        )

    def test_wer_with_speakers(self) -> None:
        """Test that WER works correctly when transcripts have speaker information."""
        # WER should ignore speaker information and only focus on words
        reference = create_transcript("hello world good morning", "A A B B")
        hypothesis = create_transcript("hello world good morning", "B B A A")

        result = self.wer(reference=reference, hypothesis=hypothesis)

        self.assertEqual(result, 0.0, "WER should be 0.0 regardless of speaker labels")

    def test_wer_components(self) -> None:
        """Test that WER components are computed correctly."""
        reference_text = "the quick brown fox"
        hypothesis_text = "the fast brown dog and cat"

        reference = create_transcript(reference_text)
        hypothesis = create_transcript(hypothesis_text)

        components = self.wer.compute_components(reference=reference, hypothesis=hypothesis)

        # Expected: 2 substitutions (quick->fast, fox->dog), 2 insertions (and, cat)
        self.assertEqual(components["num_substitutions"], 2, "Should have 2 substitutions")
        self.assertEqual(components["num_deletions"], 0, "Should have 0 deletions")
        self.assertEqual(components["num_insertions"], 2, "Should have 2 insertions")
        self.assertEqual(components["num_words"], 4, "Should have 4 reference words")

    def test_wer_components_only_deletions(self) -> None:
        """Test WER components with only deletions."""
        reference_text = "hello world good morning"
        hypothesis_text = "hello world"

        reference = create_transcript(reference_text)
        hypothesis = create_transcript(hypothesis_text)

        components = self.wer.compute_components(reference=reference, hypothesis=hypothesis)

        self.assertEqual(components["num_substitutions"], 0, "Should have 0 substitutions")
        self.assertEqual(components["num_deletions"], 2, "Should have 2 deletions")
        self.assertEqual(components["num_insertions"], 0, "Should have 0 insertions")
        self.assertEqual(components["num_words"], 4, "Should have 4 reference words")

    def test_wer_components_only_insertions(self) -> None:
        """Test WER components with only insertions."""
        reference_text = "hello world"
        hypothesis_text = "hello beautiful world today"

        reference = create_transcript(reference_text)
        hypothesis = create_transcript(hypothesis_text)

        components = self.wer.compute_components(reference=reference, hypothesis=hypothesis)

        self.assertEqual(components["num_substitutions"], 0, "Should have 0 substitutions")
        self.assertEqual(components["num_deletions"], 0, "Should have 0 deletions")
        self.assertEqual(components["num_insertions"], 2, "Should have 2 insertions")
        self.assertEqual(components["num_words"], 2, "Should have 2 reference words")

    def test_wer_components_only_substitutions(self) -> None:
        """Test WER components with only substitutions."""
        reference_text = "hello world good morning"
        hypothesis_text = "goodbye earth bad evening"

        reference = create_transcript(reference_text)
        hypothesis = create_transcript(hypothesis_text)

        components = self.wer.compute_components(reference=reference, hypothesis=hypothesis)

        self.assertEqual(components["num_substitutions"], 4, "Should have 4 substitutions")
        self.assertEqual(components["num_deletions"], 0, "Should have 0 deletions")
        self.assertEqual(components["num_insertions"], 0, "Should have 0 insertions")
        self.assertEqual(components["num_words"], 4, "Should have 4 reference words")


if __name__ == "__main__":
    unittest.main()
