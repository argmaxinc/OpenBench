# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2025 Argmax, Inc. All Rights Reserved.

import jiwer
import numpy as np
from argmaxtools.utils import get_logger
from meeteval.wer.wer.cp import CPErrorRate, cp_word_error_rate
from pyannote.metrics.base import BaseMetric
from pyannote.metrics.types import Details, MetricComponents
from pydantic import BaseModel, Field
from scipy import optimize

from ...pipeline_prediction import Transcript
from ...types import PipelineType
from ..metric import MetricOptions
from ..registry import MetricRegistry
from .text_normalizer import EnglishTextNormalizer


logger = get_logger(__name__)


class AlignmentMetrics(BaseModel):
    wer: float = Field(..., description="Word Error Rate")
    mer: float = Field(..., description="Match Error Rate")
    wil: float = Field(..., description="Word Information Loss")
    wip: float = Field(..., description="Word Information Preservation")
    hits: int = Field(..., description="Number of correct words")
    substitutions: int = Field(..., description="Number of substitutions")
    deletions: int = Field(..., description="Number of deletions")
    insertions: int = Field(..., description="Number of insertions")
    ops: list[list[jiwer.AlignmentChunk]] = Field(..., description="Alignment operations")
    truth: list[list[str]] = Field(..., description="Reference words")
    hypothesis: list[list[str]] = Field(..., description="Hypothesis words")


def parse_diarzed_words(transcript: Transcript) -> tuple[list[str], list[str] | None]:
    """Parse a list of words into text and speaker strings.

    If the transcript has no speakers, return None.
    """
    word_list = transcript.get_words()
    speaker_list = transcript.get_speakers()
    return word_list, speaker_list


class BaseWordErrorMetric(BaseMetric):
    """Base class for word error metrics."""

    def __init__(
        self,
        use_text_normalizer: bool = True,
        english_spelling_mapping: dict[str, str] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(instantaneous=True)
        self.use_text_normalizer = use_text_normalizer
        # NOTE: Currently only English text normalizer is supported
        self.text_normalizer = EnglishTextNormalizer(english_spelling_mapping)

    def _supports_paired_evaluation(self) -> bool:
        return True

    def _get_word_error_metrics(
        self, reference: Transcript, hypothesis: Transcript
    ) -> tuple[
        jiwer.AlignmentChunk,
        tuple[list[str], list[str]],
        tuple[list[str] | None, list[str] | None],
    ]:
        ref_words, ref_speakers = parse_diarzed_words(reference)
        hyp_words, hyp_speakers = parse_diarzed_words(hypothesis)

        if self.use_text_normalizer:
            ref_words, ref_speakers = self.text_normalizer(
                words=ref_words,
                speakers=ref_speakers,
            )
            hyp_words, hyp_speakers = self.text_normalizer(
                words=hyp_words,
                speakers=hyp_speakers,
            )

        result = jiwer.compute_measures(
            truth=" ".join(ref_words),
            hypothesis=" ".join(hyp_words),
        )
        result = AlignmentMetrics(**result)

        # Get alignments
        return result.ops[0], (ref_words, hyp_words), (ref_speakers, hyp_speakers)


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

        # Build cost matrix for speaker mapping
        unique_ref_speakers = sorted(set(ref_speakers))
        unique_hyp_speakers = sorted(set(hyp_speakers))
        cost_matrix = np.zeros((len(unique_ref_speakers), len(unique_hyp_speakers)))

        # Pre-process alignments to get all matching word pairs
        matching_pairs = []
        for alignment in alignments:
            if alignment.type not in ["equal", "substitute"]:
                continue
            # Get all aligned word pairs at once
            ref_indices = range(alignment.ref_start_idx, alignment.ref_end_idx)
            hyp_indices = range(alignment.hyp_start_idx, alignment.hyp_start_idx + len(ref_indices))
            matching_pairs.extend(zip(ref_indices, hyp_indices))

        # Convert to numpy arrays for faster operations
        matching_pairs = np.array(matching_pairs)
        ref_speakers_array = np.array(ref_speakers)
        hyp_speakers_array = np.array(hyp_speakers)

        # Calculate cost matrix efficiently
        for ref_idx, ref_spk in enumerate(unique_ref_speakers):
            for hyp_idx, hyp_spk in enumerate(unique_hyp_speakers):
                # Use boolean indexing to count matches
                ref_matches = ref_speakers_array[matching_pairs[:, 0]] == ref_spk
                hyp_matches = hyp_speakers_array[matching_pairs[:, 1]] == hyp_spk
                cost_matrix[ref_idx, hyp_idx] = np.sum(ref_matches & hyp_matches)

        # Find optimal speaker mapping using Hungarian algorithm
        row_ind, col_ind = optimize.linear_sum_assignment(cost_matrix, maximize=True)
        speaker_mapping = {unique_ref_speakers[i]: unique_hyp_speakers[j] for i, j in zip(row_ind, col_ind)}
        logger.debug(f"Speaker mapping: {speaker_mapping}")

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


@MetricRegistry.register_metric(
    (
        PipelineType.TRANSCRIPTION,
        PipelineType.ORCHESTRATION,
        PipelineType.STREAMING_TRANSCRIPTION,
    ),
    MetricOptions.WER,
)
class WordErrorRate(BaseWordErrorMetric):
    """Word Error Rate (WER) implementation.

    This metric evaluates the transcription accuracy at the word level.
    It uses jiwer for word alignments and calculates the standard WER metric.

    Reference:
    https://en.wikipedia.org/wiki/Word_error_rate
    """

    @classmethod
    def metric_name(cls) -> str:
        return "wer"

    @classmethod
    def metric_components(cls) -> MetricComponents:
        return [
            "num_substitutions",  # Number of word substitutions
            "num_deletions",  # Number of word deletions
            "num_insertions",  # Number of word insertions
            "num_words",  # Total number of words in reference
        ]

    def compute_components(self, reference: Transcript, hypothesis: Transcript, **kwargs) -> dict[str, int]:
        """Compute WER between reference and hypothesis.

        Args:
            reference: Reference transcript
            hypothesis: Hypothesis transcript
        """
        alignments, (ref_words, hyp_words), (_, _) = self._get_word_error_metrics(reference, hypothesis)

        # Calculate statistics
        num_substitutions = 0
        num_deletions = 0
        num_insertions = 0
        num_words = len(ref_words)

        for alignment in alignments:
            if alignment.type == "substitute":
                num_substitutions += alignment.ref_end_idx - alignment.ref_start_idx
            elif alignment.type == "delete":
                num_deletions += alignment.ref_end_idx - alignment.ref_start_idx
            elif alignment.type == "insert":
                num_insertions += alignment.hyp_end_idx - alignment.hyp_start_idx

        return {
            "num_substitutions": num_substitutions,
            "num_deletions": num_deletions,
            "num_insertions": num_insertions,
            "num_words": num_words,
        }

    def compute_metric(self, detail: Details) -> float:
        """Compute the WER metric from the components.

        WER = (S + D + I) / N
        where:
        - S is the number of substitutions
        - D is the number of deletions
        - I is the number of insertions
        - N is the total number of words in the reference
        """
        S = detail["num_substitutions"]
        D = detail["num_deletions"]
        I = detail["num_insertions"]
        N = detail["num_words"]

        return (S + D + I) / N if N > 0 else 0.0


@MetricRegistry.register_metric(PipelineType.ORCHESTRATION, MetricOptions.CPWER)
class ConcatenatedMinimumPermutationWER(BaseWordErrorMetric):
    """Concatenated minimum-Permutation Word Error Rate (cpWER) implementation.

    This metric evaluates multi-speaker transcription by:
    1. Concatenating all words per speaker (grouping by speaker ID)
    2. Finding the optimal permutation of hypothesis speakers that minimizes WER
    3. Computing WER for the best speaker permutation

    cpWER penalizes both transcription errors and speaker assignment errors, where
    a correctly transcribed word assigned to the wrong speaker is counted as an error.

    Reference:
    Watanabe, Shinji, et al. "CHiME-6 Challenge: Tackling multispeaker speech
    recognition for unsegmented recordings." arXiv preprint arXiv:2004.09249 (2020).
    """

    @classmethod
    def metric_name(cls) -> str:
        return "cpwer"

    @classmethod
    def metric_components(cls) -> MetricComponents:
        return [
            "num_substitutions",  # Number of word substitutions
            "num_deletions",  # Number of word deletions
            "num_insertions",  # Number of word insertions
            "num_words",  # Total number of words in reference
        ]

    def compute_components(self, reference: Transcript, hypothesis: Transcript, **kwargs) -> dict[str, int]:
        """Compute cpWER between reference and hypothesis.

        Args:
            reference: Reference transcript with speaker labels
            hypothesis: Hypothesis transcript with speaker labels

        Returns:
            Dictionary containing error components for the best speaker permutation
        """
        ref_words, ref_speakers = parse_diarzed_words(reference)
        hyp_words, hyp_speakers = parse_diarzed_words(hypothesis)

        # Validate that we have speaker information
        if ref_speakers is None or hyp_speakers is None:
            raise ValueError("cpWER requires speaker labels for both reference and hypothesis")

        if len(ref_words) != len(ref_speakers):
            raise ValueError(
                f"Reference words and speaker labels must have same length but got "
                f"{len(ref_words)=} and {len(ref_speakers)=}"
            )
        if len(hyp_words) != len(hyp_speakers):
            raise ValueError(
                f"Hypothesis words and speaker labels must have same length but got "
                f"{len(hyp_words)=} and {len(hyp_speakers)=}"
            )

        # Apply text normalization if enabled
        if self.use_text_normalizer:
            ref_words, ref_speakers = self.text_normalizer(
                words=ref_words,
                speakers=ref_speakers,
            )
            hyp_words, hyp_speakers = self.text_normalizer(
                words=hyp_words,
                speakers=hyp_speakers,
            )

        # Concatenate words by speaker for reference
        unique_ref_speakers = sorted(set(ref_speakers))
        ref_concatenated = {}
        for speaker in unique_ref_speakers:
            speaker_words = [word for word, spk in zip(ref_words, ref_speakers) if spk == speaker]
            ref_concatenated[speaker] = speaker_words

        # Concatenate words by speaker for hypothesis
        unique_hyp_speakers = sorted(set(hyp_speakers))
        hyp_concatenated = {}
        for speaker in unique_hyp_speakers:
            speaker_words = [word for word, spk in zip(hyp_words, hyp_speakers) if spk == speaker]
            hyp_concatenated[speaker] = speaker_words

        error_rate: CPErrorRate = cp_word_error_rate(
            reference=ref_concatenated,
            hypothesis=hyp_concatenated,
        )

        logger.debug(f"Missed Speaker: {error_rate.missed_speaker}")
        logger.debug(f"False Alarm Speaker: {error_rate.false_alarm_speaker}")
        logger.debug(f"Scored Speaker: {error_rate.scored_speaker}")
        logger.debug(f"Assigned speakers: {error_rate.assignment}")

        return {
            "num_substitutions": error_rate.substitutions,
            "num_deletions": error_rate.deletions,
            "num_insertions": error_rate.insertions,
            "num_words": error_rate.length,
        }

    def compute_metric(self, detail: Details) -> float:
        """Compute the cpWER metric from the components.

        cpWER = (S + D + I) / N
        where:
        - S is the number of substitutions (including speaker confusions)
        - D is the number of deletions
        - I is the number of insertions
        - N is the total number of words in the reference
        """
        S = detail["num_substitutions"]
        D = detail["num_deletions"]
        I = detail["num_insertions"]
        N = detail["num_words"]

        return (S + D + I) / N if N > 0 else 0.0
