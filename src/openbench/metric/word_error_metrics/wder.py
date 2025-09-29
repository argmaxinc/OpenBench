# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2025 Argmax, Inc. All Rights Reserved.

from argmaxtools.utils import get_logger
from pyannote.metrics.types import Details, MetricComponents

from ...pipeline_prediction import Transcript
from ...types import PipelineType
from ..metric import MetricOptions
from ..registry import MetricRegistry
from .speaker_matching_utils import find_optimal_speaker_mapping
from .word_error_base import BaseWordErrorMetric


logger = get_logger(__name__)


@MetricRegistry.register_metric(PipelineType.ORCHESTRATION, MetricOptions.WDER)
class WordDiarizationErrorRate(BaseWordErrorMetric):
    """Word Diarization Error Rate (WDER) implementation.

    This metric evaluates both the transcription and speaker assignment accuracy
    at the word level. It uses jiwer for word alignments and handles speaker
    mapping using the Hungarian algorithm.

    Reference:
    Shafey, Laurent El, Hagen Soltau, and Izhak Shafran.
    "Joint speech recognition and speaker diarization via sequence transduction."
    arXiv preprint arXiv:1907.05337 (2019) Equation (2).
    """

    @classmethod
    def metric_name(cls) -> str:
        return "wder"

    @classmethod
    def metric_components(cls) -> MetricComponents:
        return [
            "num_substitutions_asr",  # is the number of ASR substitutions
            "num_correct_asr",  # is the number of Correct ASR words
            "num_substitutions_asr_incorrect_speaker",  # Sis is the number of ASR Substitutions with Incorrect Speaker tokens
            "num_correct_asr_incorrect_speaker",  # Cis is the number of Correct ASR words with Incorrect Speaker tokens
        ]

    def compute_components(self, reference: Transcript, hypothesis: Transcript, **kwargs) -> dict[str, int]:
        """Compute WDER between reference and hypothesis.

        Args:
            reference: List of reference words with their speaker labels
            hypothesis: List of hypothesis words with their speaker labels
        """
        (
            alignments,
            (ref_words, hyp_words),
            (ref_speakers, hyp_speakers),
        ) = self._get_word_error_metrics(reference, hypothesis)

        if len(ref_words) != len(ref_speakers):
            raise ValueError(
                f"Reference words and speaker labels must have same length but got {ref_words=} ({len(ref_words)=}) and {ref_speakers=} ({len(ref_speakers)=})"
            )
        if len(hyp_words) != len(hyp_speakers):
            raise ValueError(
                f"Hypothesis words and speaker labels must have same length but got {hyp_words=} ({len(hyp_words)=}) and {hyp_speakers=} ({len(hyp_speakers)=})"
            )

        # Pre-process alignments to get all matching word pairs
        matching_pairs = []
        for alignment in alignments:
            if alignment.type not in ["equal", "substitute"]:
                continue
            # Get all aligned word pairs at once
            ref_indices = range(alignment.ref_start_idx, alignment.ref_end_idx)
            hyp_indices = range(alignment.hyp_start_idx, alignment.hyp_start_idx + len(ref_indices))
            matching_pairs.extend(zip(ref_indices, hyp_indices))

        # Find optimal speaker mapping using Hungarian algorithm
        speaker_mapping = find_optimal_speaker_mapping(ref_speakers, hyp_speakers, matching_pairs)

        # Calculate final statistics
        total_words = 0
        correct_assignments = 0
        num_substitutions_asr = 0
        num_substitutions_asr_incorrect_speaker = 0  # Sis
        num_correct_asr_incorrect_speaker = 0  # Cis

        for alignment in alignments:
            # We are only interested in substitutions and correct words
            if alignment.type not in ["equal", "substitute"]:
                continue

            for i in range(alignment.ref_start_idx, alignment.ref_end_idx):
                j = alignment.hyp_start_idx + (i - alignment.ref_start_idx)
                total_words += 1
                ref_spk = ref_speakers[i]
                hyp_spk = hyp_speakers[j]
                is_correct_speaker = hyp_spk == speaker_mapping.get(ref_spk)
                _type = alignment.type

                if _type == "equal":
                    correct_assignments += 1
                    num_correct_asr_incorrect_speaker += 1 if not is_correct_speaker else 0
                elif _type == "substitute":
                    num_substitutions_asr += 1
                    num_substitutions_asr_incorrect_speaker += 1 if not is_correct_speaker else 0
                else:
                    raise ValueError(f"Unknown alignment type: {_type}")

        return {
            "num_substitutions_asr": num_substitutions_asr,
            "num_correct_asr": correct_assignments,
            "num_substitutions_asr_incorrect_speaker": num_substitutions_asr_incorrect_speaker,
            "num_correct_asr_incorrect_speaker": num_correct_asr_incorrect_speaker,
        }

    def compute_metric(self, detail: Details) -> float:
        Sis = detail["num_substitutions_asr_incorrect_speaker"]
        Cis = detail["num_correct_asr_incorrect_speaker"]
        S = detail["num_substitutions_asr"]
        C = detail["num_correct_asr"]

        return (Sis + Cis) / (S + C)
