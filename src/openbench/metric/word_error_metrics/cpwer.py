# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2025 Argmax, Inc. All Rights Reserved.

import jiwer
from argmaxtools.utils import get_logger
from pyannote.metrics.types import Details, MetricComponents

from ...pipeline_prediction import Transcript
from ...types import PipelineType
from ..metric import MetricOptions
from ..registry import MetricRegistry
from .speaker_matching_utils import concatenate_transcripts_by_speaker, find_optimal_speaker_mapping
from .word_error_base import AlignmentMetrics, BaseWordErrorMetric


logger = get_logger(__name__)


@MetricRegistry.register_metric(PipelineType.ORCHESTRATION, MetricOptions.CPWER)
class ConcatenatedPermutationWordErrorRate(BaseWordErrorMetric):
    """Concatenated minimum-Permutation Word Error Rate (cpWER) implementation.

    This metric evaluates transcription accuracy after finding the optimal speaker
    assignment using the Hungarian algorithm and concatenating the transcripts
    by speaker. It computes the standard WER on the concatenated transcripts.

    The cpWER metric addresses the speaker permutation problem by:
    1. Finding optimal speaker mapping using Hungarian algorithm
    2. Concatenating reference and hypothesis transcripts by speaker
    3. Computing WER on the concatenated transcripts

    Reference:
    This metric is commonly used in speaker diarization evaluation to handle
    the speaker assignment ambiguity problem.
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
        """
        # Handle empty inputs early
        ref_words, ref_speakers = reference.get_words(), reference.get_speakers()
        hyp_words, hyp_speakers = hypothesis.get_words(), hypothesis.get_speakers()

        if not ref_words and not hyp_words:
            return {
                "num_substitutions": 0,
                "num_deletions": 0,
                "num_insertions": 0,
                "num_words": 0,
            }

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

        # Get matching pairs from alignments
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

        # Concatenate transcripts by speaker using optimal mapping
        concatenated_ref_words, concatenated_hyp_words = concatenate_transcripts_by_speaker(
            ref_words, ref_speakers, hyp_words, hyp_speakers, speaker_mapping
        )

        # Handle empty inputs
        if not concatenated_ref_words and not concatenated_hyp_words:
            return {
                "num_substitutions": 0,
                "num_deletions": 0,
                "num_insertions": 0,
                "num_words": 0,
            }

        # Compute WER on concatenated transcripts
        result = jiwer.compute_measures(
            truth=" ".join(concatenated_ref_words),
            hypothesis=" ".join(concatenated_hyp_words),
        )
        result = AlignmentMetrics(**result)

        return {
            "num_substitutions": result.substitutions,
            "num_deletions": result.deletions,
            "num_insertions": result.insertions,
            "num_words": len(concatenated_ref_words),
        }

    def compute_metric(self, detail: Details) -> float:
        """Compute the cpWER metric from the components.

        cpWER = (S + D + I) / N
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
