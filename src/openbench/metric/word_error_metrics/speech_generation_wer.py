# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2026 Argmax, Inc. All Rights Reserved.

"""Word Error Rate for speech-generation pipelines.

The metric receives a `GeneratedAudio` hypothesis (path + duration), runs
ASR on the audio to obtain a `Transcript`, and then delegates to the
standard `WordErrorRate.compute_components` against the reference. This
keeps the TTS pipeline free of ASR concerns — including from its timing
budget — while still letting WER score generated audio.

Transcription is performed via a caller-supplied `TranscriptionConfig`
(any subclass of TranscriptionConfig). The metric resolves the matching
`Pipeline` subclass through `PipelineRegistry.create_pipeline_from_config`,
so callers never have to construct the Pipeline themselves. When no
config is supplied, a default WhisperKitPro / parakeet-v2 config is used
so users on the project's default ASR don't have to wire anything up.
"""

import os
from pathlib import Path

import coremltools as ct
import librosa
from argmaxtools.utils import get_logger

from ...dataset import TranscriptionSample
from ...pipeline import Pipeline
from ...pipeline.pipeline_registry import PipelineRegistry
from ...pipeline.transcription.common import TranscriptionConfig
from ...pipeline.transcription.transcription_whisperkitpro import WhisperKitProTranscriptionConfig
from ...pipeline_prediction import GeneratedAudio, Transcript
from ...types import PipelineType
from ..metric import MetricOptions
from ..registry import MetricRegistry
from .word_error_metrics import WordErrorRate


logger = get_logger(__name__)


def _default_transcription_config() -> WhisperKitProTranscriptionConfig:
    """Build the default WhisperKitPro / parakeet-v2 transcription config.

    Pass an explicit `transcription_config=` to the metric to use a different
    backend.
    """
    cli_path = os.getenv("WHISPERKITPRO_CLI_PATH")
    if not cli_path:
        raise RuntimeError(
            "SpeechGenerationWordErrorRate's default transcription requires "
            "WHISPERKITPRO_CLI_PATH. Pass transcription_config= to the metric "
            "to use a different ASR backend."
        )

    return WhisperKitProTranscriptionConfig(
        cli_path=cli_path,
        repo_id="argmaxinc/parakeetkit-pro",
        model_variant="nvidia_parakeet-v2_476MB",
        audio_encoder_compute_units=ct.ComputeUnit.CPU_AND_NE,
        text_decoder_compute_units=ct.ComputeUnit.CPU_AND_NE,
    )


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
        transcription_config: Any `TranscriptionConfig` subclass instance.
            The matching transcription `Pipeline` is built lazily on first
            use via `PipelineRegistry.create_pipeline_from_config`. If `None`,
            a default WhisperKitPro / parakeet-v2 config is used.
        **kwargs: Forwarded to `WordErrorRate.__init__`
            (e.g. `use_text_normalizer`, `english_spelling_mapping`).
    """

    def __init__(
        self,
        transcription_config: TranscriptionConfig | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if transcription_config is not None and not isinstance(transcription_config, TranscriptionConfig):
            raise TypeError(
                "SpeechGenerationWordErrorRate.transcription_config must be a "
                f"TranscriptionConfig subclass, got {type(transcription_config).__name__}"
            )
        self._transcription_config = transcription_config
        self._pipeline: Pipeline | None = None

    def _get_pipeline(self) -> Pipeline:
        if self._pipeline is None:
            config = self._transcription_config or _default_transcription_config()
            logger.info("Building transcription pipeline for speech-generation WER: %s", type(config).__name__)
            self._pipeline = PipelineRegistry.create_pipeline_from_config(config)
        return self._pipeline

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
