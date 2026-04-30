# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2025 Argmax, Inc. All Rights Reserved.

"""Word Error Rate for speech-generation pipelines.

The metric receives a `GeneratedAudio` hypothesis (path + duration), runs
ASR on the audio to obtain a `Transcript`, and then delegates to the
standard `WordErrorRate.compute_components` against the reference. This
keeps the TTS pipeline free of ASR concerns — including from its timing
budget — while still letting WER score generated audio.

Transcription is performed by a caller-supplied callable
`transcribe_fn(audio_path: str) -> Transcript`. When not supplied a default
WhisperKitPro-backed implementation is built lazily (parakeet-v2 ASR), so
users who only need this metric with the project's default ASR don't have
to wire anything up. A future PR will plumb metric-level configuration
through the CLI; for now constructor injection is the supported swap
point.
"""

import json
import os
from pathlib import Path
from typing import Callable

import coremltools as ct
from argmaxtools.utils import get_logger

from ...engine.whisperkitpro_engine import WhisperKitPro, WhisperKitProConfig, WhisperKitProInput
from ...pipeline_prediction import GeneratedAudio, Transcript
from ...types import PipelineType
from ..metric import MetricOptions
from ..registry import MetricRegistry
from .word_error_metrics import WordErrorRate


logger = get_logger(__name__)


def _whisperkitpro_transcribe_factory(
    cli_path: str | None = None,
    repo_id: str = "argmaxinc/parakeetkit-pro",
    model_variant: str = "nvidia_parakeet-v2_476MB",
) -> Callable[[str], Transcript]:
    """Build a default `transcribe_fn` backed by the WhisperKitPro engine.

    The engine (and underlying model) is constructed once and reused
    across every call to the returned function.
    """
    cli_path = cli_path or os.getenv("WHISPERKITPRO_CLI_PATH")
    if not cli_path:
        raise RuntimeError(
            "SpeechGenerationWordErrorRate default transcription requires "
            "WHISPERKITPRO_CLI_PATH (or an explicit cli_path). Pass "
            "transcribe_fn= to the metric to use a different backend."
        )

    engine_config = WhisperKitProConfig(
        repo_id=repo_id,
        model_variant=model_variant,
        word_timestamps=True,
        chunking_strategy="vad",
        audio_encoder_compute_units=ct.ComputeUnit.CPU_AND_NE,
        text_decoder_compute_units=ct.ComputeUnit.CPU_AND_NE,
    )
    engine = WhisperKitPro(cli_path=cli_path, transcription_config=engine_config)

    def transcribe(audio_path: str) -> Transcript:
        engine_output = engine(WhisperKitProInput(audio_path=Path(audio_path), keep_audio=False))

        json_path = engine_output.json_report_path
        try:
            if not json_path.exists():
                raise RuntimeError(f"Transcription report not found at {json_path}")
            with json_path.open("r") as f:
                data = json.load(f)

            words, starts, ends = [], [], []
            for seg in data.get("segments", []):
                for w in seg.get("words", []):
                    words.append(w["word"])
                    if "start" in w:
                        starts.append(w["start"])
                    if "end" in w:
                        ends.append(w["end"])

            return Transcript.from_words_info(
                words=words,
                start=starts if starts else None,
                end=ends if ends else None,
            )
        finally:
            json_path.unlink(missing_ok=True)
            srt_path = engine_output.srt_report_path
            if srt_path is not None:
                Path(srt_path).unlink(missing_ok=True)

    return transcribe


@MetricRegistry.register_metric(PipelineType.SPEECH_GENERATION, MetricOptions.WER)
class SpeechGenerationWordErrorRate(WordErrorRate):
    """WER metric for speech-generation pipelines.

    Args:
        transcribe_fn: Callable taking an audio file path and returning a
            `Transcript`. If `None`, a default WhisperKitPro+parakeet-v2
            transcriber is built lazily on first use.
        **kwargs: Forwarded to `WordErrorRate.__init__`
            (e.g. `use_text_normalizer`, `english_spelling_mapping`).
    """

    def __init__(
        self,
        transcribe_fn: Callable[[str], Transcript] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._explicit_transcribe = transcribe_fn
        self._default_transcribe: Callable[[str], Transcript] | None = None

    def _get_transcribe(self) -> Callable[[str], Transcript]:
        if self._explicit_transcribe is not None:
            return self._explicit_transcribe
        if self._default_transcribe is None:
            logger.info("Building default WhisperKitPro transcriber for speech-generation WER")
            self._default_transcribe = _whisperkitpro_transcribe_factory()
        return self._default_transcribe

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
        hypothesis_transcript = self._get_transcribe()(hypothesis.audio_path)
        logger.info("TTS WER hypothesis transcript: " + hypothesis_transcript.get_transcript_string()[:120] + "...")
        return super().compute_components(reference, hypothesis_transcript, **kwargs)
