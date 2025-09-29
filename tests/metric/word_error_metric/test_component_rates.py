# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2025 Argmax, Inc. All Rights Reserved.

import unittest

from openbench.metric.word_error_metrics.wer import DeletionRate, InsertionRate, SubstitutionRate
from openbench.pipeline_prediction import Transcript


def create_transcript(text: str) -> Transcript:
    """Create a transcript from text string."""
    return Transcript.from_words_info(
        words=text.split(),
        start=None,
        end=None,
        speaker=None,
    )


class TestSubstitutionRate(unittest.TestCase):
    def setUp(self) -> None:
        self.substitution_rate = SubstitutionRate()

    def tearDown(self) -> None:
        self.substitution_rate = None

    def test_perfect_match(self) -> None:
        """Test substitution rate with perfect match."""
        reference_text = "hello world how are you"
        hypothesis_text = "hello world how are you"

        reference = create_transcript(reference_text)
        hypothesis = create_transcript(hypothesis_text)

        result = self.substitution_rate(reference=reference, hypothesis=hypothesis)
        self.assertEqual(result, 0.0, f"Expected 0.0, got {result}")

    def test_substitutions(self) -> None:
        """Test substitution rate with substitutions."""
        reference_text = "hello world how are you"
        hypothesis_text = "hello there how are you"  # 1 substitution

        reference = create_transcript(reference_text)
        hypothesis = create_transcript(hypothesis_text)

        result = self.substitution_rate(reference=reference, hypothesis=hypothesis)
        expected = 1.0 / 5.0  # 1 substitution out of 5 words
        self.assertAlmostEqual(result, expected, places=5, msg=f"Expected {expected}, got {result}")

    def test_empty_inputs(self) -> None:
        """Test substitution rate with empty inputs."""
        reference_text = ""
        hypothesis_text = ""

        reference = create_transcript(reference_text)
        hypothesis = create_transcript(hypothesis_text)

        result = self.substitution_rate(reference=reference, hypothesis=hypothesis)
        self.assertEqual(result, 0.0, f"Expected 0.0 for empty inputs, got {result}")


class TestDeletionRate(unittest.TestCase):
    def setUp(self) -> None:
        self.deletion_rate = DeletionRate()

    def tearDown(self) -> None:
        self.deletion_rate = None

    def test_perfect_match(self) -> None:
        """Test deletion rate with perfect match."""
        reference_text = "hello world how are you"
        hypothesis_text = "hello world how are you"

        reference = create_transcript(reference_text)
        hypothesis = create_transcript(hypothesis_text)

        result = self.deletion_rate(reference=reference, hypothesis=hypothesis)
        self.assertEqual(result, 0.0, f"Expected 0.0, got {result}")

    def test_deletions(self) -> None:
        """Test deletion rate with deletions."""
        reference_text = "hello world how are you"
        hypothesis_text = "hello world how"  # 2 deletions

        reference = create_transcript(reference_text)
        hypothesis = create_transcript(hypothesis_text)

        result = self.deletion_rate(reference=reference, hypothesis=hypothesis)
        expected = 1.0 / 5.0  # 1 deletion out of 5 words
        self.assertAlmostEqual(result, expected, places=5, msg=f"Expected {expected}, got {result}")

    def test_empty_inputs(self) -> None:
        """Test deletion rate with empty inputs."""
        reference_text = ""
        hypothesis_text = ""

        reference = create_transcript(reference_text)
        hypothesis = create_transcript(hypothesis_text)

        result = self.deletion_rate(reference=reference, hypothesis=hypothesis)
        self.assertEqual(result, 0.0, f"Expected 0.0 for empty inputs, got {result}")


class TestInsertionRate(unittest.TestCase):
    def setUp(self) -> None:
        self.insertion_rate = InsertionRate()

    def tearDown(self) -> None:
        self.insertion_rate = None

    def test_perfect_match(self) -> None:
        """Test insertion rate with perfect match."""
        reference_text = "hello world how are you"
        hypothesis_text = "hello world how are you"

        reference = create_transcript(reference_text)
        hypothesis = create_transcript(hypothesis_text)

        result = self.insertion_rate(reference=reference, hypothesis=hypothesis)
        self.assertEqual(result, 0.0, f"Expected 0.0, got {result}")

    def test_insertions(self) -> None:
        """Test insertion rate with insertions."""
        reference_text = "hello world how are you"
        hypothesis_text = "hello world how are you today"  # 1 insertion

        reference = create_transcript(reference_text)
        hypothesis = create_transcript(hypothesis_text)

        result = self.insertion_rate(reference=reference, hypothesis=hypothesis)
        expected = 1.0 / 5.0  # 1 insertion out of 5 reference words
        self.assertAlmostEqual(result, expected, places=5, msg=f"Expected {expected}, got {result}")

    def test_empty_inputs(self) -> None:
        """Test insertion rate with empty inputs."""
        reference_text = ""
        hypothesis_text = ""

        reference = create_transcript(reference_text)
        hypothesis = create_transcript(hypothesis_text)

        result = self.insertion_rate(reference=reference, hypothesis=hypothesis)
        self.assertEqual(result, 0.0, f"Expected 0.0 for empty inputs, got {result}")


if __name__ == "__main__":
    unittest.main()
