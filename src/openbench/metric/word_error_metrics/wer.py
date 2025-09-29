# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2025 Argmax, Inc. All Rights Reserved.

from argmaxtools.utils import get_logger
from pyannote.metrics.types import Details, MetricComponents

from ...pipeline_prediction import Transcript
from ...types import PipelineType
from ..metric import MetricOptions
from ..registry import MetricRegistry
from .word_error_base import BaseWordErrorMetric


logger = get_logger(__name__)


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
                num_substitutions += 1
            elif alignment.type == "delete":
                num_deletions += 1
            elif alignment.type == "insert":
                num_insertions += 1

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


@MetricRegistry.register_metric(
    (
        PipelineType.TRANSCRIPTION,
        PipelineType.ORCHESTRATION,
        PipelineType.STREAMING_TRANSCRIPTION,
    ),
    MetricOptions.SUBSTITUTION_RATE,
)
class SubstitutionRate(BaseWordErrorMetric):
    """Substitution Rate implementation.

    This metric evaluates word substitution errors as a rate (S/N).
    It uses jiwer for word alignments and calculates the substitution rate.

    Reference:
    https://en.wikipedia.org/wiki/Word_error_rate
    """

    @classmethod
    def metric_name(cls) -> str:
        return "substitution_rate"

    @classmethod
    def metric_components(cls) -> MetricComponents:
        return [
            "num_substitutions",  # Number of word substitutions
            "num_words",  # Total number of words in reference
        ]

    def compute_components(self, reference: Transcript, hypothesis: Transcript, **kwargs) -> dict[str, int]:
        """Compute substitution rate between reference and hypothesis.

        Args:
            reference: Reference transcript
            hypothesis: Hypothesis transcript
        """
        # Handle empty inputs early
        ref_words = reference.get_words()
        if not ref_words:
            return {
                "num_substitutions": 0,
                "num_words": 0,
            }

        alignments, (ref_words, hyp_words), (_, _) = self._get_word_error_metrics(reference, hypothesis)

        # Calculate statistics
        num_substitutions = 0
        num_words = len(ref_words)

        for alignment in alignments:
            if alignment.type == "substitute":
                num_substitutions += 1

        return {
            "num_substitutions": num_substitutions,
            "num_words": num_words,
        }

    def compute_metric(self, detail: Details) -> float:
        """Compute the substitution rate from the components.

        Substitution Rate = S / N
        where:
        - S is the number of substitutions
        - N is the total number of words in the reference
        """
        S = detail["num_substitutions"]
        N = detail["num_words"]

        return S / N if N > 0 else 0.0


@MetricRegistry.register_metric(
    (
        PipelineType.TRANSCRIPTION,
        PipelineType.ORCHESTRATION,
        PipelineType.STREAMING_TRANSCRIPTION,
    ),
    MetricOptions.DELETION_RATE,
)
class DeletionRate(BaseWordErrorMetric):
    """Deletion Rate implementation.

    This metric evaluates word deletion errors as a rate (D/N).
    It uses jiwer for word alignments and calculates the deletion rate.

    Reference:
    https://en.wikipedia.org/wiki/Word_error_rate
    """

    @classmethod
    def metric_name(cls) -> str:
        return "deletion_rate"

    @classmethod
    def metric_components(cls) -> MetricComponents:
        return [
            "num_deletions",  # Number of word deletions
            "num_words",  # Total number of words in reference
        ]

    def compute_components(self, reference: Transcript, hypothesis: Transcript, **kwargs) -> dict[str, int]:
        """Compute deletion rate between reference and hypothesis.

        Args:
            reference: Reference transcript
            hypothesis: Hypothesis transcript
        """
        # Handle empty inputs early
        ref_words = reference.get_words()
        if not ref_words:
            return {
                "num_deletions": 0,
                "num_words": 0,
            }

        alignments, (ref_words, hyp_words), (_, _) = self._get_word_error_metrics(reference, hypothesis)

        # Calculate statistics
        num_deletions = 0
        num_words = len(ref_words)

        for alignment in alignments:
            if alignment.type == "delete":
                num_deletions += 1

        return {
            "num_deletions": num_deletions,
            "num_words": num_words,
        }

    def compute_metric(self, detail: Details) -> float:
        """Compute the deletion rate from the components.

        Deletion Rate = D / N
        where:
        - D is the number of deletions
        - N is the total number of words in the reference
        """
        D = detail["num_deletions"]
        N = detail["num_words"]

        return D / N if N > 0 else 0.0


@MetricRegistry.register_metric(
    (
        PipelineType.TRANSCRIPTION,
        PipelineType.ORCHESTRATION,
        PipelineType.STREAMING_TRANSCRIPTION,
    ),
    MetricOptions.INSERTION_RATE,
)
class InsertionRate(BaseWordErrorMetric):
    """Insertion Rate implementation.

    This metric evaluates word insertion errors as a rate (I/N).
    It uses jiwer for word alignments and calculates the insertion rate.

    Reference:
    https://en.wikipedia.org/wiki/Word_error_rate
    """

    @classmethod
    def metric_name(cls) -> str:
        return "insertion_rate"

    @classmethod
    def metric_components(cls) -> MetricComponents:
        return [
            "num_insertions",  # Number of word insertions
            "num_words",  # Total number of words in reference
        ]

    def compute_components(self, reference: Transcript, hypothesis: Transcript, **kwargs) -> dict[str, int]:
        """Compute insertion rate between reference and hypothesis.

        Args:
            reference: Reference transcript
            hypothesis: Hypothesis transcript
        """
        # Handle empty inputs early
        ref_words = reference.get_words()
        if not ref_words:
            return {
                "num_insertions": 0,
                "num_words": 0,
            }

        alignments, (ref_words, hyp_words), (_, _) = self._get_word_error_metrics(reference, hypothesis)

        # Calculate statistics
        num_insertions = 0
        num_words = len(ref_words)

        for alignment in alignments:
            if alignment.type == "insert":
                num_insertions += 1

        return {
            "num_insertions": num_insertions,
            "num_words": num_words,
        }

    def compute_metric(self, detail: Details) -> float:
        """Compute the insertion rate from the components.

        Insertion Rate = I / N
        where:
        - I is the number of insertions
        - N is the total number of words in the reference
        """
        I = detail["num_insertions"]
        N = detail["num_words"]

        return I / N if N > 0 else 0.0
