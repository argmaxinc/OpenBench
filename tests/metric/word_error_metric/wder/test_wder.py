# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2025 Argmax, Inc. All Rights Reserved.

import json
import unittest

from huggingface_hub import hf_hub_download

from openbench.metric import WordDiarizationErrorRate
from openbench.pipeline_prediction import Transcript


TEST_FIXTURES_REPO = "argmaxinc/test-fixtures"


def create_transcript(text: str, speakers: str) -> Transcript:
    """Create a list of Words from text and speaker strings."""
    return Transcript.from_words_info(
        words=text.split(),
        start=None,
        end=None,
        speaker=speakers.split(),
    )


class TestWDER(unittest.TestCase):
    def setUp(self) -> None:
        self.wder = WordDiarizationErrorRate()

    def tearDown(self) -> None:
        self.wder = None

    def _compute_and_compare(
        self,
        reference_text: str,
        reference_speakers: str,
        hypothesis_text: str,
        hypothesis_speakers: str,
        expected_wder: float,
    ) -> None:
        reference = create_transcript(reference_text, reference_speakers)
        hypothesis = create_transcript(hypothesis_text, hypothesis_speakers)

        result = self.wder(reference=reference, hypothesis=hypothesis)
        self.assertAlmostEqual(
            result,
            expected_wder,
            places=4,
            msg=f"WDER result: {result}, Expected: {expected_wder}",
        )

    def test_wder_perfect_match(self) -> None:
        # Reference text-speakers:
        reference_text = "a b c d e f g h i j"
        reference_speakers = "1 1 1 1 2 2 2 2 2"
        # Hypothesis text-speakers:
        hypothesis_text = "a b c d e f g h i j"
        hypothesis_speakers = "1 1 1 1 2 2 2 2 2"

        self._compute_and_compare(
            reference_text=reference_text,
            reference_speakers=reference_speakers,
            hypothesis_text=hypothesis_text,
            hypothesis_speakers=hypothesis_speakers,
            expected_wder=0.0,
        )

    def test_wder_completely_wrong_speaker(self) -> None:
        # Reference text-speakers:
        reference_text = "a b c d e f g h i j"
        reference_speakers = "1 1 1 1 2 2 2 2 2"
        # Hypothesis text-speakers:
        hypothesis_text = "a b c d e f g h i j"
        hypothesis_speakers = "3 3 3 3 3 3 3 3 3 3"

        self._compute_and_compare(
            reference_text=reference_text,
            reference_speakers=reference_speakers,
            hypothesis_text=hypothesis_text,
            hypothesis_speakers=hypothesis_speakers,
            expected_wder=0.44444,
        )

    def test_wder_completely_wrong_text(self) -> None:
        # Reference text-speakers:
        reference_text = "a b c d e f g h i j"
        reference_speakers = "1 1 1 1 2 2 2 2 2"
        # Hypothesis text-speakers:
        hypothesis_text = "l l l l l l l l l l"
        hypothesis_speakers = "1 1 1 1 2 2 2 2 2"

        self._compute_and_compare(
            reference_text=reference_text,
            reference_speakers=reference_speakers,
            hypothesis_text=hypothesis_text,
            hypothesis_speakers=hypothesis_speakers,
            expected_wder=0.0,
        )

    def test_wder_partially_wrong_text(self) -> None:
        # Reference text-speakers:
        reference_text = "a b c d e f g h i j"
        reference_speakers = "1 1 1 1 2 2 2 2 2"
        # Hypothesis text-speakers:
        hypothesis_text = "a b c d x f g h i j"
        hypothesis_speakers = "1 1 1 1 2 2 2 2 2"

        self._compute_and_compare(
            reference_text=reference_text,
            reference_speakers=reference_speakers,
            hypothesis_text=hypothesis_text,
            hypothesis_speakers=hypothesis_speakers,
            expected_wder=0.0,
        )

    def test_wder_partially_wrong_speaker(self) -> None:
        # Reference text-speakers:
        reference_text = "a b c d e f g h i j"
        reference_speakers = "1 1 1 1 2 2 2 2 2"
        # Hypothesis text-speakers:
        hypothesis_text = "a b c d e f g h i j"
        hypothesis_speakers = "1 1 1 1 3 3 3 3 3"

        self._compute_and_compare(
            reference_text=reference_text,
            reference_speakers=reference_speakers,
            hypothesis_text=hypothesis_text,
            hypothesis_speakers=hypothesis_speakers,
            expected_wder=0.0,
        )

    def test_wder_text_deletion(self) -> None:
        # Reference text-speakers:
        reference_text = "a b c d e f g h i j"
        reference_speakers = "1 1 1 1 2 2 2 2 2"
        # Hypothesis text-speakers:
        hypothesis_text = "a b c d e f g h"
        hypothesis_speakers = "1 1 1 1 2 2 2"

        self._compute_and_compare(
            reference_text=reference_text,
            reference_speakers=reference_speakers,
            hypothesis_text=hypothesis_text,
            hypothesis_speakers=hypothesis_speakers,
            expected_wder=0.0,
        )

    def test_wder_text_insertion(self) -> None:
        # Reference text-speakers:
        reference_text = "a b c d e f g h i j"
        reference_speakers = "1 1 1 1 2 2 2 2 2"
        # Hypothesis text-speakers:
        hypothesis_text = "a b c d e f g h i j k"
        hypothesis_speakers = "1 1 1 1 2 2 2 2 2"

        self._compute_and_compare(
            reference_text=reference_text,
            reference_speakers=reference_speakers,
            hypothesis_text=hypothesis_text,
            hypothesis_speakers=hypothesis_speakers,
            expected_wder=0.0,
        )

    def test_wder_less_speakers(self) -> None:
        # Reference text-speakers:
        reference_text = "a b c d e f g h i j"
        reference_speakers = "1 1 1 1 2 2 3 4 5"
        # Hypothesis text-speakers:
        hypothesis_text = "a b c d e f g h i j"
        hypothesis_speakers = "1 1 1 1 2 2 2 2 2"

        self._compute_and_compare(
            reference_text=reference_text,
            reference_speakers=reference_speakers,
            hypothesis_text=hypothesis_text,
            hypothesis_speakers=hypothesis_speakers,
            expected_wder=0.33333,
        )

    def test_wder_more_speakers(self) -> None:
        # Reference text-speakers:
        reference_text = "a b c d e f g h i j"
        reference_speakers = "1 1 1 1 2 2 2 2 2"
        # Hypothesis text-speakers:
        hypothesis_text = "a b c d e f g h i j"
        hypothesis_speakers = "1 1 1 1 2 2 2 2 2 3"

        self._compute_and_compare(
            reference_text=reference_text,
            reference_speakers=reference_speakers,
            hypothesis_text=hypothesis_text,
            hypothesis_speakers=hypothesis_speakers,
            expected_wder=0.0,
        )

    def test_wder_real_more_hyp_speakers(self) -> None:
        """Test WDER with a real example with more hypothesis speakers than reference."""
        json_path = hf_hub_download(
            repo_id=TEST_FIXTURES_REPO,
            repo_type="dataset",
            filename="more_hyp_speakers.json",
            subfolder="tests/cpwer",
        )
        with open(json_path, "r") as f:
            data = json.load(f)

        ref_text = data["reference"]["text"]
        ref_speakers = data["reference"]["speakers"]
        hyp_text = data["hypothesis"]["text"]
        hyp_speakers = data["hypothesis"]["speakers"]

        self._compute_and_compare(
            reference_text=ref_text,
            reference_speakers=ref_speakers,
            hypothesis_text=hyp_text,
            hypothesis_speakers=hyp_speakers,
            expected_wder=0.08333,
        )

    def test_wder_real_fewer_hyp_speakers(self) -> None:
        """Test WDER with a real example with more hypothesis speakers than reference."""
        json_path = hf_hub_download(
            repo_id=TEST_FIXTURES_REPO,
            repo_type="dataset",
            filename="fewer_hyp_speakers.json",
            subfolder="tests/cpwer",
        )
        with open(json_path, "r") as f:
            data = json.load(f)

        ref_text = data["reference"]["text"]
        ref_speakers = data["reference"]["speakers"]
        hyp_text = data["hypothesis"]["text"]
        hyp_speakers = data["hypothesis"]["speakers"]

        self._compute_and_compare(
            reference_text=ref_text,
            reference_speakers=ref_speakers,
            hypothesis_text=hyp_text,
            hypothesis_speakers=hyp_speakers,
            expected_wder=0.16635,
        )
