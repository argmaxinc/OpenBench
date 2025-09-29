# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2025 Argmax, Inc. All Rights Reserved.

import unittest

from openbench.metric import ConcatenatedPermutationWordErrorRate
from openbench.pipeline_prediction import Transcript


def create_transcript(text: str, speakers: str) -> Transcript:
    """Create a list of Words from text and speaker strings."""
    return Transcript.from_words_info(
        words=text.split(),
        start=None,
        end=None,
        speaker=speakers.split(),
    )


class TestCPWER(unittest.TestCase):
    def setUp(self) -> None:
        self.cpwer = ConcatenatedPermutationWordErrorRate()

    def tearDown(self) -> None:
        self.cpwer = None

    def test_cpwer_perfect_match(self) -> None:
        """Test cpWER with perfect match."""
        reference_text = "a b c d e f g h i j"
        reference_speakers = "1 1 1 1 2 2 2 2 2 2"
        hypothesis_text = "a b c d e f g h i j"
        hypothesis_speakers = "1 1 1 1 2 2 2 2 2 2"

        reference = create_transcript(reference_text, reference_speakers)
        hypothesis = create_transcript(hypothesis_text, hypothesis_speakers)

        result = self.cpwer(reference=reference, hypothesis=hypothesis)
        self.assertEqual(result, 0.0, f"Expected 0.0, got {result}")

    def test_cpwer_speaker_permutation(self) -> None:
        """Test cpWER with speaker permutation (should be handled correctly)."""
        reference_text = "a b c d e f g h i j"
        reference_speakers = "1 1 1 1 2 2 2 2 2 2"
        hypothesis_text = "a b c d e f g h i j"
        hypothesis_speakers = "2 2 2 2 1 1 1 1 1 1"  # Swapped speakers

        reference = create_transcript(reference_text, reference_speakers)
        hypothesis = create_transcript(hypothesis_text, hypothesis_speakers)

        result = self.cpwer(reference=reference, hypothesis=hypothesis)
        self.assertEqual(result, 0.0, f"Expected 0.0 for speaker permutation, got {result}")

    def test_cpwer_text_errors(self) -> None:
        """Test cpWER with text errors."""
        reference_text = "a b c d e f g h i j"
        reference_speakers = "1 1 1 1 2 2 2 2 2 2"
        hypothesis_text = "a b x d e f g h i j"  # One substitution
        hypothesis_speakers = "1 1 1 1 2 2 2 2 2 2"

        reference = create_transcript(reference_text, reference_speakers)
        hypothesis = create_transcript(hypothesis_text, hypothesis_speakers)

        result = self.cpwer(reference=reference, hypothesis=hypothesis)
        expected = 1.0 / 10.0  # 1 error out of 10 words
        self.assertAlmostEqual(result, expected, places=5, msg=f"Expected {expected}, got {result}")

    def test_cpwer_deletions(self) -> None:
        """Test cpWER with deletions."""
        reference_text = "a b c d e f g h i j"
        reference_speakers = "1 1 1 1 2 2 2 2 2 2"
        hypothesis_text = "a b c d e f g h"  # Missing last 2 words
        hypothesis_speakers = "1 1 1 1 2 2 2 2"

        reference = create_transcript(reference_text, reference_speakers)
        hypothesis = create_transcript(hypothesis_text, hypothesis_speakers)

        result = self.cpwer(reference=reference, hypothesis=hypothesis)
        expected = 2.0 / 10.0  # 2 deletions out of 10 words
        self.assertAlmostEqual(result, expected, places=5, msg=f"Expected {expected}, got {result}")

    def test_cpwer_insertions(self) -> None:
        """Test cpWER with insertions."""
        reference_text = "a b c d e f g h i j"
        reference_speakers = "1 1 1 1 2 2 2 2 2 2"
        hypothesis_text = "a b c d e f g h i j k l"  # 2 extra words
        hypothesis_speakers = "1 1 1 1 2 2 2 2 2 2 1 1"

        reference = create_transcript(reference_text, reference_speakers)
        hypothesis = create_transcript(hypothesis_text, hypothesis_speakers)

        result = self.cpwer(reference=reference, hypothesis=hypothesis)
        expected = 2.0 / 10.0  # 2 insertions out of 10 reference words
        self.assertAlmostEqual(result, expected, places=5, msg=f"Expected {expected}, got {result}")

    def test_cpwer_complex_case(self) -> None:
        """Test cpWER with complex speaker assignment and text errors."""
        reference_text = "hello world how are you today"
        reference_speakers = "1 1 2 2 2 2"
        hypothesis_text = "hello world how are you today"
        hypothesis_speakers = "2 2 1 1 1 1"  # Swapped speakers

        reference = create_transcript(reference_text, reference_speakers)
        hypothesis = create_transcript(hypothesis_text, hypothesis_speakers)

        result = self.cpwer(reference=reference, hypothesis=hypothesis)
        self.assertEqual(result, 0.0, f"Expected 0.0 for speaker permutation, got {result}")

    def test_cpwer_empty_inputs(self) -> None:
        """Test cpWER with empty inputs."""
        reference_text = ""
        reference_speakers = ""
        hypothesis_text = ""
        hypothesis_speakers = ""

        reference = create_transcript(reference_text, reference_speakers)
        hypothesis = create_transcript(hypothesis_text, hypothesis_speakers)

        result = self.cpwer(reference=reference, hypothesis=hypothesis)
        self.assertEqual(result, 0.0, f"Expected 0.0 for empty inputs, got {result}")


if __name__ == "__main__":
    unittest.main()
