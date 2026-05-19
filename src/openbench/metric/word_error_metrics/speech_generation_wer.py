# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2025 Argmax, Inc. All Rights Reserved.

"""Word Error Rate for speech-generation pipelines.

The metric receives a `GeneratedAudio` hypothesis (path + duration), runs
ASR on the audio to obtain a `Transcript`, and then delegates to the
standard `WordErrorRate.compute_components` against the reference. This
keeps the TTS pipeline free of ASR concerns — including from its timing
budget — while still letting WER score generated audio.

Transcription is performed by a caller-supplied `TranscriptionPipeline`
(any pipeline registered with `PipelineType.TRANSCRIPTION`). When not
supplied, a default WhisperKitPro / parakeet-v2 pipeline is built lazily
on first use so users on the project's default ASR don't have to wire
anything up. A future PR will plumb metric-level configuration through
the CLI; for now constructor injection is the supported swap point.
"""

import os
from pathlib import Path

import coremltools as ct
import librosa
from argmaxtools.utils import get_logger

from ...dataset import TranscriptionSample
from ...pipeline import Pipeline
from ...pipeline.transcription.transcription_whisperkitpro import (
    WhisperKitProTranscriptionConfig,
    WhisperKitProTranscriptionPipeline,
)
from ...pipeline_prediction import GeneratedAudio, Transcript
from ...types import PipelineType
from ..metric import MetricOptions
from ..registry import MetricRegistry
from .word_error_metrics import WordErrorRate


logger = get_logger(__name__)


def _build_default_transcription_pipeline() -> Pipeline:
    """Build the default WhisperKitPro / parakeet-v2 transcription pipeline.

    This is the project's default ASR for evaluating TTS output; pass an
    explicit `transcription_pipeline=` to the metric to use a different
    backend.
    """
    cli_path = os.getenv("WHISPERKITPRO_CLI_PATH")
    if not cli_path:
        raise RuntimeError(
            "SpeechGenerationWordErrorRate's default transcription requires "
            "WHISPERKITPRO_CLI_PATH. Pass transcription_pipeline= to the metric "
            "to use a different ASR backend."
        )

    config = WhisperKitProTranscriptionConfig(
        cli_path=cli_path,
        repo_id="argmaxinc/parakeetkit-pro",
        model_variant="nvidia_parakeet-v2_476MB",
        audio_encoder_compute_units=ct.ComputeUnit.CPU_AND_NE,
        text_decoder_compute_units=ct.ComputeUnit.CPU_AND_NE,
    )
    return WhisperKitProTranscriptionPipeline(config)


def _build_transcription_sample(audio_path: str) -> TranscriptionSample:
    """Wrap a WAV on disk in a TranscriptionSample so a pipeline can consume it.

    The reference is a no-op `Transcript` — transcription pipelines only
    read it via `parse_input` for fields like keyword boosting (which we
    aren't using here), not as ground truth.
    """
    waveform, sample_rate = librosa.load(audio_path, sr=None)
    return TranscriptionSample(
        audio_name=Path(audio_path).stem,
        waveform=waveform,
        sample_rate=int(sample_rate),
        reference=Transcript(words=[]),
        extra_info={},
    )


@MetricRegistry.register_metric(PipelineType.SPEECH_GENERATION, MetricOptions.WER)
class SpeechGenerationWordErrorRate(WordErrorRate):
    """WER metric for speech-generation pipelines.

    Args:
        transcription_pipeline: Any TRANSCRIPTION-typed `Pipeline`. Its
            `__call__(sample) -> PipelineOutput[Transcript]` is invoked
            on the generated audio. If `None`, a default WhisperKitPro /
            parakeet-v2 pipeline is built lazily on first use.
        **kwargs: Forwarded to `WordErrorRate.__init__`
            (e.g. `use_text_normalizer`, `english_spelling_mapping`).
    """

    def __init__(
        self,
        transcription_pipeline: Pipeline | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if transcription_pipeline is not None and transcription_pipeline.pipeline_type != PipelineType.TRANSCRIPTION:
            raise ValueError(
                "SpeechGenerationWordErrorRate.transcription_pipeline must have "
                f"pipeline_type=TRANSCRIPTION, got {transcription_pipeline.pipeline_type}"
            )
        self._explicit_pipeline = transcription_pipeline
        self._default_pipeline: Pipeline | None = None

    def _get_pipeline(self) -> Pipeline:
        if self._explicit_pipeline is not None:
            return self._explicit_pipeline
        if self._default_pipeline is None:
            logger.info(
                "Building default WhisperKitPro / parakeet-v2 transcription pipeline for speech-generation WER"
            )
            self._default_pipeline = _build_default_transcription_pipeline()
        return self._default_pipeline

    def compute_components(self, reference: Transcript, hypothesis: GeneratedAudio, **kwargs) -> dict[str, int]:
        """Transcribe the generated audio, then delegate to WER.

        Args:
            reference: Reference transcript built from the original prompt.
            hypothesis: Generated audio (path + duration) produced by a TTS pipeline.
        """
        if not isinstance(hypothesis, GeneratedAudio):
            raise TypeError(
                f"SpeechGenerationWordErrorRate expected hypothesis of type GeneratedAudio, "
                f"got {type(hypothesis).__name__}"
            )
        sample = _build_transcription_sample(hypothesis.audio_path)
        pipeline_output = self._get_pipeline()(sample)
        hypothesis_transcript: Transcript = pipeline_output.prediction
        logger.info("TTS WER hypothesis transcript: " + hypothesis_transcript.get_transcript_string()[:120] + "...")
        return super().compute_components(reference, hypothesis_transcript, **kwargs)
