# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2025 Argmax, Inc. All Rights Reserved.

import warnings

import jiwer
import scipy.stats
from argmaxtools.utils import get_logger
from pyannote.metrics.base import BaseMetric
from pyannote.metrics.types import Details, MetricComponents
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

from ...pipeline_prediction import StreamingTranscript, Transcript
from ...types import PipelineType
from ..metric import MetricOptions
from ..registry import MetricRegistry


logger = get_logger(__name__)
NUM_DELETIONS = "num_deletions"
NUM_SUBSTITUTIONS = "num_substitutions"
NUM_INSERTIONS = "num_insertions"

normalizer = BasicTextNormalizer()


class BaseNumCorrections(BaseMetric):
    """Metric Calculation"""

    def compute_num_corrections(self, interim_results, correction_type):
        if interim_results is None:
            return None
        n_deletion = 0
        n_subs = 0
        n_insertion = 0
        for f in range(len(interim_results) - 1):
            out = jiwer.process_words(normalizer(interim_results[f + 1]), normalizer(interim_results[f]))
            prev_len = len(out.hypotheses[0])
            for alig in out.alignments[0]:
                if alig.type == "delete" and alig.ref_start_idx < prev_len:
                    n_deletion += alig.ref_end_idx - alig.ref_start_idx
                elif alig.type == "substitute":
                    n_subs += alig.ref_end_idx - alig.ref_start_idx
                elif alig.type == "insert":
                    n_insertion += (alig.hyp_end_idx - alig.hyp_start_idx)
        if correction_type == "insertion":
            # Intentionally Flipped
            return n_deletion
        elif correction_type == "substitutions":
            return n_subs
        elif correction_type == "deletion":
            # Intentionally Flipped
            return n_insertion

    def _supports_paired_evaluation(self) -> bool:
        return True

    def confidence_interval(self, alpha: float = 0.9) -> tuple[float, tuple[float, float]]:
        if self.results_[0][0] == "NA":
            return None, (None, None)

        values = [r[self.metric_name_] for _, r in self.results_]

        if len(values) == 0:
            raise ValueError("Please evaluate a bunch of files before computing confidence interval.")

        elif len(values) == 1:
            warnings.warn("Cannot compute a reliable confidence interval out of just one file.")
            center = lower = upper = values[0]
            return center, (lower, upper)

        else:
            return scipy.stats.bayes_mvs(values, alpha=alpha)[0]

    def __call__(
        self,
        reference: Transcript,
        hypothesis: StreamingTranscript,
        detailed: bool = False,
        uri: str | None = None,
        **kwargs,
    ):
        # compute metric components
        components = self.compute_components(reference, hypothesis, **kwargs)

        # compute rate based on components
        components[self.metric_name_] = self.compute_metric(components)

        # keep track of this computation
        uri = uri or getattr(reference, "uri", "NA")
        self.results_.append((uri, components))

        # accumulate components
        for name in self.components_:
            if components[name] is not None:
                self.accumulated_[name] += components[name]
            else:
                self.accumulated_[name] = None
                break

        if detailed:
            return components

        return components[self.metric_name_]


# NOTE: reference is not used in this metric since this is computed from hypothesis.interim_results
@MetricRegistry.register_metric(PipelineType.STREAMING_TRANSCRIPTION, MetricOptions.NUM_DELETIONS)
class NumDeletions(BaseNumCorrections):
    """Metric Calculation"""

    @classmethod
    def metric_name(cls):
        return NUM_DELETIONS

    @classmethod
    def metric_components(cls) -> MetricComponents:
        return [NUM_DELETIONS]

    def compute_components(self, reference: Transcript, hypothesis: StreamingTranscript, **kwargs) -> Details:
        (num_corrections) = self.compute_num_corrections(hypothesis.interim_results, "deletion")

        detail = {NUM_DELETIONS: num_corrections}

        return detail

    def compute_metric(self, detail: Details | None) -> float:
        if detail[NUM_DELETIONS] is None:
            return None
        return detail[NUM_DELETIONS]


# NOTE: reference is not used in this metric since this is computed from hypothesis.interim_results
@MetricRegistry.register_metric(PipelineType.STREAMING_TRANSCRIPTION, MetricOptions.NUM_SUBSTITUTIONS)
class NumSubstitutions(BaseNumCorrections):
    """Metric Calculation"""

    @classmethod
    def metric_name(cls):
        return NUM_SUBSTITUTIONS

    @classmethod
    def metric_components(cls) -> MetricComponents:
        return [NUM_SUBSTITUTIONS]

    def compute_components(self, reference: Transcript, hypothesis: StreamingTranscript, **kwargs) -> Details:
        (num_corrections) = self.compute_num_corrections(hypothesis.interim_results, "substitutions")

        if num_corrections is None:
            return {
                NUM_SUBSTITUTIONS: None,
            }

        detail = {NUM_SUBSTITUTIONS: num_corrections}

        return detail

    def compute_metric(self, detail: Details | None) -> float:
        if detail[NUM_SUBSTITUTIONS] is None:
            return None
        return detail[NUM_SUBSTITUTIONS]


# NOTE: reference is not used in this metric since this is computed from hypothesis.interim_results
@MetricRegistry.register_metric(PipelineType.STREAMING_TRANSCRIPTION, MetricOptions.NUM_INSERTIONS)
class NumInsertions(BaseNumCorrections):
    """Metric Calculation"""

    @classmethod
    def metric_name(cls):
        return NUM_INSERTIONS

    @classmethod
    def metric_components(cls) -> MetricComponents:
        return [NUM_INSERTIONS]

    def compute_components(self, reference: Transcript, hypothesis: StreamingTranscript, **kwargs) -> Details:
        (num_corrections) = self.compute_num_corrections(hypothesis.interim_results, "insertion")

        if num_corrections is None:
            return {
                NUM_INSERTIONS: None,
            }

        detail = {NUM_INSERTIONS: num_corrections}

        return detail

    def compute_metric(self, detail: Details | None) -> float:
        if detail[NUM_INSERTIONS] is None:
            return None
        return detail[NUM_INSERTIONS]
