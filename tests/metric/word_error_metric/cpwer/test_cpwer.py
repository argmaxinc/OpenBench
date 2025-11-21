# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2025 Argmax, Inc. All Rights Reserved.

import json
import random
import unittest

from argmaxtools.utils import get_logger
from huggingface_hub import hf_hub_download

from openbench.metric import ConcatenatedMinimumPermutationWER
from openbench.pipeline_prediction import Transcript


logger = get_logger(__name__)

RANDOM_SEED = 42
TEST_FIXTURES_REPO = "argmaxinc/test-fixtures"


def create_transcript(text: str, speakers: str) -> Transcript:
    """Create a Transcript from text and speaker strings."""
    return Transcript.from_words_info(
        words=text.split(),
        start=None,
        end=None,
        speaker=speakers.split(),
    )


class TestCPWER(unittest.TestCase):
    def setUp(self) -> None:
        self.cpwer = ConcatenatedMinimumPermutationWER(use_text_normalizer=False)

    def tearDown(self) -> None:
        self.cpwer = None

    def _compute_cpwer(
        self,
        reference_text: str,
        reference_speakers: str,
        hypothesis_text: str,
        hypothesis_speakers: str,
        detailed: bool = False,
    ) -> float:
        reference = create_transcript(reference_text, reference_speakers)
        hypothesis = create_transcript(hypothesis_text, hypothesis_speakers)

        result = self.cpwer(reference=reference, hypothesis=hypothesis, detailed=detailed)
        return result

    def test_cpwer_perfect_match(self) -> None:
        """Test cpWER with perfect transcription and speaker labels."""
        # Reference text-speakers:
        reference_text = "hello world good morning"
        reference_speakers = "A A B B"
        # Hypothesis text-speakers:
        hypothesis_text = "hello world good morning"
        hypothesis_speakers = "A A B B"

        result = self._compute_cpwer(
            reference_text=reference_text,
            reference_speakers=reference_speakers,
            hypothesis_text=hypothesis_text,
            hypothesis_speakers=hypothesis_speakers,
        )
        self.assertEqual(result, 0.0, "Perfect match should have cpWER of 0.0")

    def test_cpwer_permuted_speakers(self) -> None:
        """Test cpWER with correct transcription but permuted speaker labels."""
        # Reference text-speakers:
        reference_text = "hello world good morning"
        reference_speakers = "A A B B"
        # Hypothesis text-speakers (speakers swapped):
        hypothesis_text = "hello world good morning"
        hypothesis_speakers = "B B A A"

        result = self._compute_cpwer(
            reference_text=reference_text,
            reference_speakers=reference_speakers,
            hypothesis_text=hypothesis_text,
            hypothesis_speakers=hypothesis_speakers,
        )
        self.assertEqual(
            result,
            0.0,
            "cpWER should be 0.0 when only speaker labels are permuted",
        )

    def test_cpwer_single_substitution(self) -> None:
        """Test cpWER with a single word substitution."""
        # Reference text-speakers:
        reference_text = "hello world good morning"
        reference_speakers = "A A B B"
        # Hypothesis text-speakers (one word wrong):
        hypothesis_text = "hello world bad morning"
        hypothesis_speakers = "A A B B"

        result = self._compute_cpwer(
            reference_text=reference_text,
            reference_speakers=reference_speakers,
            hypothesis_text=hypothesis_text,
            hypothesis_speakers=hypothesis_speakers,
        )
        self.assertAlmostEqual(
            result,
            0.25,
            places=4,
            msg="cpWER should be 0.25 (1 substitution / 4 words)",
        )

    def test_cpwer_single_deletion(self) -> None:
        """Test cpWER with a single word deletion."""
        # Reference text-speakers:
        reference_text = "hello world good morning"
        reference_speakers = "A A B B"
        # Hypothesis text-speakers (one word deleted):
        hypothesis_text = "hello world good"
        hypothesis_speakers = "A A B"

        result = self._compute_cpwer(
            reference_text=reference_text,
            reference_speakers=reference_speakers,
            hypothesis_text=hypothesis_text,
            hypothesis_speakers=hypothesis_speakers,
        )
        self.assertAlmostEqual(
            result,
            0.25,
            places=4,
            msg="cpWER should be 0.25 (1 deletion / 4 words)",
        )

    def test_cpwer_single_insertion(self) -> None:
        """Test cpWER with a single word insertion."""
        # Reference text-speakers:
        reference_text = "hello world good morning"
        reference_speakers = "A A B B"
        # Hypothesis text-speakers (one word inserted):
        hypothesis_text = "hello world good morning friend"
        hypothesis_speakers = "A A B B B"

        result = self._compute_cpwer(
            reference_text=reference_text,
            reference_speakers=reference_speakers,
            hypothesis_text=hypothesis_text,
            hypothesis_speakers=hypothesis_speakers,
        )
        self.assertAlmostEqual(
            result,
            0.25,
            places=4,
            msg="cpWER should be 0.25 (1 insertion / 4 words)",
        )

    def test_cpwer_completely_wrong_text(self) -> None:
        """Test cpWER with completely incorrect transcription."""
        # Reference text-speakers:
        reference_text = "hello world good morning"
        reference_speakers = "A A B B"
        # Hypothesis text-speakers (all words wrong):
        hypothesis_text = "foo bar baz qux"
        hypothesis_speakers = "A A B B"

        result = self._compute_cpwer(
            reference_text=reference_text,
            reference_speakers=reference_speakers,
            hypothesis_text=hypothesis_text,
            hypothesis_speakers=hypothesis_speakers,
        )
        self.assertEqual(result, 1.0, "cpWER should be 1.0 when all words are wrong")

    def test_cpwer_empty_hypothesis(self) -> None:
        """Test cpWER with empty hypothesis."""
        # Reference text-speakers:
        reference_text = "hello world good morning"
        reference_speakers = "A A B B"
        # Hypothesis text-speakers (empty - no words means no speakers):
        hypothesis_text = "a"
        hypothesis_speakers = "X"

        result = self._compute_cpwer(
            reference_text=reference_text,
            reference_speakers=reference_speakers,
            hypothesis_text=hypothesis_text,
            hypothesis_speakers=hypothesis_speakers,
        )
        # 3 deletions, 1 substitution = 4 errors / 4 words = 1.0
        self.assertEqual(result, 1.0, "cpWER should be 1.0 when almost all is deleted")

    def test_cpwer_more_hypothesis_speakers(self) -> None:
        """Test cpWER when hypothesis has more speakers than reference (all separate)."""
        # Reference text-speakers:
        reference_text = "hello world good morning"
        reference_speakers = "A A A A"
        # Hypothesis text-speakers (more speakers, each word has unique speaker):
        hypothesis_text = "hello world good morning"
        hypothesis_speakers = "X Y Z W"

        result = self._compute_cpwer(
            reference_text=reference_text,
            reference_speakers=reference_speakers,
            hypothesis_text=hypothesis_text,
            hypothesis_speakers=hypothesis_speakers,
        )
        # In cpWER, when we concatenate by speaker:
        # Ref: A="hello world good morning" (4 words)
        # Hyp: X="hello", Y="world", Z="good", W="morning" (1 word each)
        # Best permutation: match one hyp speaker to ref speaker A
        # This results in 3 deletions + 3 insertions = 6 errors / 4 words = 1.5
        self.assertAlmostEqual(
            result,
            1.5,
            places=2,
            msg="cpWER should be 1.5 when speakers are completely fragmented",
        )

    def test_cpwer_fewer_hypothesis_speakers(self) -> None:
        """Test cpWER when hypothesis has fewer speakers than reference."""
        # Reference text-speakers:
        reference_text = "hello world good morning"
        reference_speakers = "A B C D"
        # Hypothesis text-speakers (all merged into one speaker):
        hypothesis_text = "hello world good morning"
        hypothesis_speakers = "X X X X"

        result = self._compute_cpwer(
            reference_text=reference_text,
            reference_speakers=reference_speakers,
            hypothesis_text=hypothesis_text,
            hypothesis_speakers=hypothesis_speakers,
        )
        # In cpWER, when we concatenate by speaker:
        # Ref: A="hello", B="world", C="good", D="morning" (1 word each)
        # Hyp: X="hello world good morning" (4 words)
        # Best permutation: match hyp speaker X to one ref speaker
        # This results in 3 deletions + 3 insertions = 6 errors / 4 words = 1.5
        self.assertAlmostEqual(
            result,
            1.5,
            places=2,
            msg="cpWER should be 1.5 when all speakers are merged",
        )

    def test_cpwer_mixed_errors(self) -> None:
        """Test cpWER with mixed substitution, insertion, and deletion errors."""
        # Reference text-speakers:
        reference_text = "the quick brown fox jumps over the lazy dog"
        reference_speakers = "A A A B B B C C C"
        # Hypothesis text-speakers (mixed errors):
        hypothesis_text = "the fast brown fox jumping over lazy dog end"
        hypothesis_speakers = "X X X Y Y Y Z Z Z"

        result = self._compute_cpwer(
            reference_text=reference_text,
            reference_speakers=reference_speakers,
            hypothesis_text=hypothesis_text,
            hypothesis_speakers=hypothesis_speakers,
        )
        # Expected: 2 substitutions (quick->fast, jumps->jumping),
        #           1 deletion (the), 1 insertion (end)
        # Total errors: 4, Total words: 9
        # cpWER = 4/9 ≈ 0.444
        self.assertAlmostEqual(
            result,
            4 / 9,
            places=3,
            msg="cpWER should be ~0.444 with mixed errors",
        )

    def test_cpwer_three_speakers_permutation(self) -> None:
        """Test cpWER with three speakers that need optimal permutation."""
        # Reference text-speakers:
        reference_text = "apple banana cherry date elderberry fig"
        reference_speakers = "A A B B C C"
        # Hypothesis text-speakers (speakers rotated):
        hypothesis_text = "apple banana cherry date elderberry fig"
        hypothesis_speakers = "C C A A B B"

        result = self._compute_cpwer(
            reference_text=reference_text,
            reference_speakers=reference_speakers,
            hypothesis_text=hypothesis_text,
            hypothesis_speakers=hypothesis_speakers,
        )
        self.assertEqual(
            result,
            0.0,
            "cpWER should be 0.0 with correct transcription and rotated speakers",
        )

    def test_cpwer_many_speakers(self) -> None:
        """Test cpWER with many speakers (testing Hungarian algorithm path)."""
        # Reference text-speakers (8 speakers, triggers Hungarian algorithm):
        words = ["w" + str(i) for i in range(16)]
        speakers = ["S" + str(i % 8) for i in range(16)]
        reference_text = " ".join(words)
        reference_speakers = " ".join(speakers)

        # Hypothesis with permuted speaker labels
        hyp_speaker_mapping = {
            "S0": "H7",
            "S1": "H6",
            "S2": "H5",
            "S3": "H4",
            "S4": "H3",
            "S5": "H2",
            "S6": "H1",
            "S7": "H0",
        }
        hyp_speakers = [hyp_speaker_mapping[s] for s in speakers]
        hypothesis_text = " ".join(words)
        hypothesis_speakers = " ".join(hyp_speakers)

        result = self._compute_cpwer(
            reference_text=reference_text,
            reference_speakers=reference_speakers,
            hypothesis_text=hypothesis_text,
            hypothesis_speakers=hypothesis_speakers,
        )
        self.assertEqual(
            result,
            0.0,
            "cpWER should be 0.0 with correct transcription and many permuted speakers",
        )

    def test_cpwer_random_modifications(self) -> None:
        """Test cpWER with random modifications to verify stability."""
        # Reference text-speakers:
        reference_text = "the quick brown fox jumps over the lazy dog"
        reference_speakers = "A A A B B B C C C"

        # Set random seed for reproducibility
        random.seed(RANDOM_SEED)

        # Run multiple iterations
        for _ in range(5):
            words = reference_text.split()
            speakers = reference_speakers.split()
            hypothesis_words = words.copy()
            hypothesis_speakers = speakers.copy()

            # Randomly modify some words and speakers
            for pos in range(len(hypothesis_words)):
                if random.random() < 0.3:  # 30% chance to modify
                    modification_type = random.choice(["text", "speaker"])

                    if modification_type == "text":
                        hypothesis_words[pos] = "error"
                    elif modification_type == "speaker":
                        hypothesis_speakers[pos] = random.choice(["X", "Y", "Z"])

            hypothesis_text = " ".join(hypothesis_words)
            hypothesis_speakers = " ".join(hypothesis_speakers)

            result = self._compute_cpwer(
                reference_text=reference_text,
                reference_speakers=reference_speakers,
                hypothesis_text=hypothesis_text,
                hypothesis_speakers=hypothesis_speakers,
            )

            # Just verify it doesn't crash and returns a valid value
            self.assertGreaterEqual(result, 0.0, "cpWER should be >= 0.0")
            self.assertLessEqual(result, 2.0, "cpWER should be <= 2.0 in practice")

    def test_cpwer_real_more_hyp_speakers(self) -> None:
        """Test cpWER with a real example with more hypothesis speakers than reference."""
        json_path = hf_hub_download(
            repo_id=TEST_FIXTURES_REPO,
            repo_type="dataset",
            filename="more_hyp_speakers.json",
            subfolder="tests/cpwer",
        )
        with open(json_path, "r") as f:
            data = json.load(f)

        expected_results = data["expected_results"]
        result = self._compute_cpwer(
            reference_text=data["reference"]["text"],
            reference_speakers=data["reference"]["speakers"],
            hypothesis_text=data["hypothesis"]["text"],
            hypothesis_speakers=data["hypothesis"]["speakers"],
            detailed=True,
        )

        for key, value in expected_results.items():
            self.assertAlmostEqual(result[key], value, places=4, msg=f"{key} should be {value}, but got {result[key]}")

    def test_cpwer_real_fewer_hyp_speakers(self) -> None:
        """Test cpWER with a real example with fewer hypothesis speakers than reference."""
        json_path = hf_hub_download(
            repo_id=TEST_FIXTURES_REPO,
            repo_type="dataset",
            filename="fewer_hyp_speakers.json",
            subfolder="tests/cpwer",
        )
        with open(json_path, "r") as f:
            data = json.load(f)

        expected_results = data["expected_results"]
        result = self._compute_cpwer(
            reference_text=data["reference"]["text"],
            reference_speakers=data["reference"]["speakers"],
            hypothesis_text=data["hypothesis"]["text"],
            hypothesis_speakers=data["hypothesis"]["speakers"],
            detailed=True,
        )

        for key, value in expected_results.items():
            self.assertAlmostEqual(result[key], value, places=4, msg=f"{key} should be {value}, but got {result[key]}")


if __name__ == "__main__":
    unittest.main()
