# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2025 Argmax, Inc. All Rights Reserved.

import string
import warnings

import jiwer
import numpy as np
import scipy.stats
from argmaxtools.utils import get_logger
from pyannote.metrics.base import BaseMetric
from pyannote.metrics.types import Details, MetricComponents
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

from ...pipeline_prediction import StreamingTranscript, Transcript, Word
from ...types import PipelineType
from ..metric import MetricOptions
from ..registry import MetricRegistry


normalizer = BasicTextNormalizer()

logger = get_logger(__name__)
STREAMING_LATENCY = "streaming_latency"
AVG_LAT = "scaled_avg_latency"
AUD_DUR = "audio_duration"
CONFIRMED_STREAMING_LATENCY = "confirmed_streaming_latency"
CONFIRMED_AVG_LAT = "confirmed_scaled_avg_latency"
CONFIRMED_AUD_DUR = "confirmed_audio_duration"
MODEL_STREAMING_LATENCY = "model_timestamp_streaming_latency"
MODEL_AVG_LAT = "model_timestamp_scaled_avg_latency"
MODEL_AUD_DUR = "model_timestamp_audio_duration"
MODEL_CONFIRMED_STREAMING_LATENCY = "model_timestamp_confirmed_streaming_latency"
MODEL_CONFIRMED_AVG_LAT = "model_timestamp_confirmed_scaled_avg_latency"
MODEL_CONFIRMED_AUD_DUR = "model_timestamp_confirmed_audio_duration"

# Latency metrics are adapted from the example at:
# https://developers.deepgram.com/docs/measuring-streaming-latency
# with slight modifications to the original metric definitions.


class BaseStreamingLatency(BaseMetric):
    """Metric Calculation
    The Average Reported Latency when an interim result t received, is computed as:

    Min Latency = audio_cursor[t] - transcript_cursor[t]
    Max Latency = audio_cursor[t] - transcript_cursor[t-1]
    Average Latency = (Min Latency + Max Latency) / 2

    audio_cursor[t]: Total amount of audio (in ms) sent by the time interim result t is received.
    transcript_cursor[t]: Total amount of audio (in ms) transcribed when interim result t is received, including
    the content in result t.
    """

    def map_hypot_idx_to_ref_idx(self, hypot_idx: int, alignment_chunks: list[jiwer.AlignmentChunk]) -> int:
        for chunk in alignment_chunks:
            if chunk.hyp_start_idx <= hypot_idx < chunk.hyp_end_idx:
                offset = hypot_idx - chunk.hyp_start_idx
                return chunk.ref_start_idx + offset

    def compute_min_max_latency(
        self,
        interim_results: list[str],
        audio_cursor: list[float],
        words: list[Word],
        model_timestamps: list[list[dict[str, float]]],
        model_timestamps_based: bool = False,
    ):
        # If API doesn't support Hypothesis/Confirmed text return None
        if interim_results is None or audio_cursor is None:
            return None, None, None
        # If API doesn't support word timestamps return None
        if (model_timestamps is None) and model_timestamps_based:
            return None, None, None

        transcript_gt = " ".join(word.word for word in words)
        transcript_cursor_gt = []
        gt_min_latency_l = []
        gt_max_latency_l = []
        gt_audio_duration = []
        if model_timestamps_based:
            model_min_latency_l = []
            model_max_latency_l = []
            transcript_cursor_model = []
            model_audio_duration = []

        transcript_gt = transcript_gt.translate(str.maketrans("", "", string.punctuation))
        for l in range(len(interim_results)):
            out = jiwer.process_words(normalizer(transcript_gt), normalizer(interim_results[l]))
            alignments_l = [out.alignments[0][i].type for i in range(len(out.alignments[0]))]

            indices = [i for i, val in enumerate(alignments_l) if val == "equal"]

            if indices:
                first_index = indices[0]
                last_index = indices[-1]
                ref_start = out.alignments[0][first_index].ref_start_idx
                ref_end = out.alignments[0][last_index].ref_end_idx
                if l == 0:
                    start_timestamp = words[ref_start].start
                else:
                    if (normalizer(interim_results[l - 1]) == " ") or (normalizer(interim_results[l]) == " "):
                        continue
                    # Find the updated segment
                    out_diff = jiwer.process_words(
                        normalizer(interim_results[l - 1]),
                        normalizer(interim_results[l]),
                    )
                    alignments_types = [out_diff.alignments[0][i].type for i in range(len(out_diff.alignments[0]))]
                    indices_diff = [
                        i for i, val in enumerate(alignments_types) if val == "insert" or val == "substitute"
                    ]
                    if indices_diff:
                        diff_ref_start = out_diff.alignments[0][indices_diff[0]].hyp_start_idx
                        # Map Start idx of updated segment to GT Transcript idx
                        actual_idx = self.map_hypot_idx_to_ref_idx(diff_ref_start, out.alignments[0])
                        try:
                            start_timestamp = words[actual_idx].start
                        # TODO: Handle Edge Cases
                        except Exception:
                            continue
                    else:
                        # Current behaviour is when there is no new word,
                        # exclude this interim result from latency calculation
                        continue

                end_timestamp = words[ref_end - 1].end
                gt_audio_duration.append(end_timestamp - start_timestamp)
                if model_timestamps_based:
                    start_timestamp_model = model_timestamps[l][0]["start"]
                    end_timestamp_model = model_timestamps[l][-1]["end"]
                    transcript_cursor_model.append(end_timestamp_model)
                    model_audio_duration.append(end_timestamp_model - start_timestamp_model)

                transcript_cursor_gt.append(start_timestamp + (end_timestamp - start_timestamp))
                gt_min_latency_l.append(audio_cursor[l] - transcript_cursor_gt[-1])
                if model_timestamps_based:
                    model_min_latency_l.append(audio_cursor[l] - transcript_cursor_model[-1])
                if len(transcript_cursor_gt) < 2:
                    gt_max_latency_l.append(audio_cursor[l])
                    if model_timestamps_based:
                        model_max_latency_l.append(audio_cursor[l])
                else:
                    gt_max_latency_l.append(audio_cursor[l] - transcript_cursor_gt[-2])
                    if model_timestamps_based:
                        model_max_latency_l.append(audio_cursor[l] - transcript_cursor_model[-2])
                # Exclude negative latency values.
                # A negative latency indicates that the system predicted the transcript
                # before receiving the entire corresponding audio. This is an edge case.
                # Map those latencies to zero
                gt_max_latency_l[-1] = 0 if gt_max_latency_l[-1] < 0 else gt_max_latency_l[-1]
                gt_min_latency_l[-1] = 0 if gt_min_latency_l[-1] < 0 else gt_min_latency_l[-1]
                if model_timestamps_based:
                    model_max_latency_l[-1] = 0 if model_max_latency_l[-1] < 0 else model_max_latency_l[-1]
                    model_min_latency_l[-1] = 0 if model_min_latency_l[-1] < 0 else model_min_latency_l[-1]

                logger.debug("Min GT Latency: " + str(gt_min_latency_l[-1]))
                logger.debug("Max GT Latency: " + str(gt_max_latency_l[-1]))
                if model_timestamps_based:
                    logger.debug("Model Min Latency: " + str(model_min_latency_l[-1]))
                    logger.debug("Model Max Latency: " + str(model_max_latency_l[-1]))

        if model_timestamps_based:
            return model_min_latency_l, model_max_latency_l, model_audio_duration
        else:
            return gt_min_latency_l, gt_max_latency_l, gt_audio_duration

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
    ) -> Details | float:
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


@MetricRegistry.register_metric(PipelineType.STREAMING_TRANSCRIPTION, MetricOptions.STREAMING_LATENCY)
class StreamingLatency(BaseStreamingLatency):
    """Metric Calculation
    The Average Reported Latency when an interim result t received, is computed as:

    Min Latency = audio_cursor[t] - transcript_cursor[t]
    Max Latency = audio_cursor[t] - transcript_cursor[t-1]
    Average Latency = (Min Latency + Max Latency) / 2

    audio_cursor[t]: Total amount of audio (in ms) sent by the time interim result t is received.
    transcript_cursor[t]: Total amount of audio (in ms) transcribed when interim result t is received, including
    the content in result t.
    """

    @classmethod
    def metric_name(cls) -> str:
        return STREAMING_LATENCY

    @classmethod
    def metric_components(cls) -> MetricComponents:
        return [AVG_LAT, AUD_DUR]

    def compute_components(self, reference: Transcript, hypothesis: StreamingTranscript, **kwargs) -> Details:
        (
            gt_min_latency_l,
            gt_max_latency_l,
            gt_audio_duration,
        ) = self.compute_min_max_latency(
            hypothesis.interim_results, hypothesis.audio_cursor, reference.words, None
        )  # GT Timestamp Based
        if gt_max_latency_l is None or gt_min_latency_l is None:
            return {AVG_LAT: None, AUD_DUR: 0}
        avg_latency = [(min + max) / 2 for min, max in zip(gt_min_latency_l, gt_max_latency_l)]
        detail = {
            AVG_LAT: sum(avg_lat * aud_dur for avg_lat, aud_dur in zip(avg_latency, gt_audio_duration)),
            AUD_DUR: np.sum(gt_audio_duration),
        }

        return detail

    def compute_metric(self, detail: Details | None) -> float:
        if detail[AUD_DUR] == 0:
            return None
        return detail[AVG_LAT] / detail[AUD_DUR]


@MetricRegistry.register_metric(PipelineType.STREAMING_TRANSCRIPTION, MetricOptions.CONFIRMED_STREAMING_LATENCY)
class ConfirmedStreamingLatency(BaseStreamingLatency):
    @classmethod
    def metric_name(cls) -> str:
        return CONFIRMED_STREAMING_LATENCY

    @classmethod
    def metric_components(cls) -> MetricComponents:
        return [CONFIRMED_AVG_LAT, CONFIRMED_AUD_DUR]

    def compute_components(self, reference: Transcript, hypothesis: StreamingTranscript, **kwargs) -> Details:
        (
            gt_min_latency_l,
            gt_max_latency_l,
            gt_audio_duration,
        ) = super().compute_min_max_latency(
            hypothesis.confirmed_interim_results,
            hypothesis.confirmed_audio_cursor,
            reference.words,
            None,
        )  # GT Timestamp Based
        if gt_max_latency_l is None or gt_min_latency_l is None:
            return {CONFIRMED_AVG_LAT: None, CONFIRMED_AUD_DUR: 0}

        avg_latency = [(min + max) / 2 for min, max in zip(gt_min_latency_l, gt_max_latency_l)]
        detail = {
            CONFIRMED_AVG_LAT: sum(avg_lat * aud_dur for avg_lat, aud_dur in zip(avg_latency, gt_audio_duration)),
            CONFIRMED_AUD_DUR: np.sum(gt_audio_duration),
        }

        return detail

    def compute_metric(self, detail: Details | None) -> float:
        if detail[CONFIRMED_AUD_DUR] == 0:
            return None
        return detail[CONFIRMED_AVG_LAT] / detail[CONFIRMED_AUD_DUR]


@MetricRegistry.register_metric(
    PipelineType.STREAMING_TRANSCRIPTION,
    MetricOptions.MODELTIMESTAMP_CONFIRMED_STRM_LATENCY,
)
class ModelTimestampBasedConfirmedStreamingLatency(BaseStreamingLatency):
    @classmethod
    def metric_name(cls) -> str:
        return MODEL_STREAMING_LATENCY

    @classmethod
    def metric_components(cls) -> MetricComponents:
        return [MODEL_CONFIRMED_AVG_LAT, MODEL_CONFIRMED_AUD_DUR]

    def compute_components(self, reference: Transcript, hypothesis: StreamingTranscript, **kwargs) -> Details:
        (
            model_min_latency_l,
            model_max_latency_l,
            model_audio_duration,
        ) = self.compute_min_max_latency(
            hypothesis.confirmed_interim_results,
            hypothesis.confirmed_audio_cursor,
            reference.words,
            hypothesis.model_timestamps_confirmed,
            model_timestamps_based=True,
        )
        if model_min_latency_l is None or model_max_latency_l is None:
            return {MODEL_CONFIRMED_AVG_LAT: None, MODEL_CONFIRMED_AUD_DUR: 0}

        avg_latency = [(min + max) / 2 for min, max in zip(model_min_latency_l, model_max_latency_l)]
        detail = {
            MODEL_CONFIRMED_AVG_LAT: sum(
                avg_lat * aud_dur for avg_lat, aud_dur in zip(avg_latency, model_audio_duration)
            ),
            MODEL_CONFIRMED_AUD_DUR: np.sum(model_audio_duration),
        }

        return detail

    def compute_metric(self, detail: Details | None) -> float:
        if detail[MODEL_CONFIRMED_AUD_DUR] == 0:
            return None
        return detail[MODEL_CONFIRMED_AVG_LAT] / detail[MODEL_CONFIRMED_AUD_DUR]


@MetricRegistry.register_metric(PipelineType.STREAMING_TRANSCRIPTION, MetricOptions.MODELTIMESTAMP_STREAMING_LATENCY)
class ModelTimestampBasedStreamingLatency(BaseStreamingLatency):
    @classmethod
    def metric_name(cls) -> str:
        return MODEL_STREAMING_LATENCY

    @classmethod
    def metric_components(cls) -> MetricComponents:
        return [MODEL_AVG_LAT, MODEL_AUD_DUR]

    def compute_components(self, reference: Transcript, hypothesis: StreamingTranscript, **kwargs) -> Details:
        (
            model_min_latency_l,
            model_max_latency_l,
            model_audio_duration,
        ) = super().compute_min_max_latency(
            hypothesis.interim_results,
            hypothesis.audio_cursor,
            reference.words,
            hypothesis.model_timestamps_hypothesis,
            model_timestamps_based=True,
        )
        if model_min_latency_l is None or model_max_latency_l is None:
            return {MODEL_AVG_LAT: None, MODEL_AUD_DUR: 0}

        avg_latency = [(min + max) / 2 for min, max in zip(model_min_latency_l, model_max_latency_l)]
        detail = {
            MODEL_AVG_LAT: sum(avg_lat * aud_dur for avg_lat, aud_dur in zip(avg_latency, model_audio_duration)),
            MODEL_AUD_DUR: np.sum(model_audio_duration),
        }

        return detail

    def compute_metric(self, detail: Details | None) -> float:
        if detail[MODEL_AUD_DUR] == 0:
            return None
        return detail[MODEL_AVG_LAT] / detail[MODEL_AUD_DUR]
