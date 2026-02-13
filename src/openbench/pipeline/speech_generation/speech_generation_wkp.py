# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2025 Argmax, Inc. All Rights Reserved.

"""
Speech generation pipeline using WhisperKit CLI.

Generates TTS audio from text prompts, then transcribes
the generated audio back to text for WER evaluation
against the original prompt.
"""

import json
import subprocess
import time
from pathlib import Path
from typing import Callable

from argmaxtools.utils import get_logger
from pydantic import BaseModel, Field

from ...dataset.dataset_base import BaseSample
from ...dataset.dataset_speech_generation import (
    SpeechGenerationSample,
)
from ...engine.whisperkitpro_engine import (
    WhisperKitPro,
    WhisperKitProConfig,
    WhisperKitProInput,
)
from ...pipeline_prediction import Transcript
from ..base import (
    Pipeline,
    PipelineOutput,
    PipelineType,
    register_pipeline,
)
from .common import SpeechGenerationConfig, SpeechGenerationOutput


logger = get_logger(__name__)

TEMP_TTS_AUDIO_DIR = Path("./temp_tts_audio")


class SpeechGenerationInput(BaseModel):
    """Input for the speech generation pipeline."""

    text: str = Field(
        ...,
        description="Text prompt to generate speech from.",
    )
    audio_name: str = Field(
        ...,
        description=("Unique identifier for this sample (used for temp file naming)."),
    )


@register_pipeline
class WhisperKitSpeechGenerationPipeline(Pipeline):
    """Speech generation pipeline using WhisperKit CLI.

    This pipeline:
    1. Generates audio from text via whisperkit-cli tts
    2. Transcribes audio via WhisperKitPro engine
    3. Returns transcription as Transcript for WER eval
    4. Cleans up temporary audio and report files
    """

    _config_class = SpeechGenerationConfig
    pipeline_type = PipelineType.SPEECH_GENERATION

    def build_pipeline(
        self,
    ) -> Callable[[SpeechGenerationInput], Transcript]:
        config = self.config
        pipeline_ref = self

        # Build the WhisperKitPro engine for transcription
        # (downloads model once, reuses for all samples)
        transcription_engine = self._build_transcription_engine()

        def generate_and_transcribe(
            input: SpeechGenerationInput,
        ) -> Transcript:
            TEMP_TTS_AUDIO_DIR.mkdir(parents=True, exist_ok=True)

            audio_path = TEMP_TTS_AUDIO_DIR / f"{input.audio_name}.wav"

            # -- Step 1: Generate audio via TTS --
            tts_cmd = [
                config.cli_path,
                "tts",
                "--text",
                input.text,
                "--speaker",
                config.speaker,
                "--language",
                config.language,
                "--output-path",
                str(audio_path),
                "--temperature",
                str(config.temperature),
                "--top-k",
                str(config.top_k),
                "--max-new-tokens",
                str(config.max_new_tokens),
            ]

            if config.seed is not None:
                tts_cmd.extend(["--seed", str(config.seed)])
            if config.models_path is not None:
                tts_cmd.extend(["--models-path", config.models_path])
            if config.model_repo is not None:
                tts_cmd.extend(["--model-repo", config.model_repo])
            if config.version_dir is not None:
                tts_cmd.extend(["--version-dir", config.version_dir])
            if config.tokenizer is not None:
                tts_cmd.extend(["--tokenizer", config.tokenizer])

            logger.debug(f"Running TTS: {' '.join(tts_cmd)}")

            tts_result = subprocess.run(tts_cmd, capture_output=True, text=True)

            if tts_result.returncode != 0:
                raise RuntimeError(
                    "whisperkit-cli tts failed "
                    f"(exit {tts_result.returncode}):\n"
                    f"  stdout: "
                    f"{tts_result.stdout[:500]}\n"
                    f"  stderr: "
                    f"{tts_result.stderr[:500]}"
                )

            if not audio_path.exists():
                raise RuntimeError(f"TTS completed but audio file not found at {audio_path}")

            logger.info(f"Generated TTS audio: {audio_path}")

            # -- Step 2: Read audio duration before
            #    transcription (engine may delete file) --
            try:
                import soundfile as sf

                info = sf.info(str(audio_path))
                pipeline_ref._last_generated_duration = info.duration
            except Exception as e:
                logger.warning(f"WAV duration read failed: {e}")
                pipeline_ref._last_generated_duration = None

            # -- Step 3: Transcribe via WhisperKitPro --
            engine_input = WhisperKitProInput(
                audio_path=audio_path,
                keep_audio=False,
            )
            engine_output = transcription_engine(engine_input)

            # -- Step 4: Parse transcription report --
            json_path = engine_output.json_report_path
            if json_path.exists():
                with json_path.open("r") as f:
                    data = json.load(f)
                all_words, all_starts, all_ends = (
                    [],
                    [],
                    [],
                )
                for seg in data.get("segments", []):
                    for w in seg.get("words", []):
                        all_words.append(w["word"])
                        if "start" in w:
                            all_starts.append(w["start"])
                        if "end" in w:
                            all_ends.append(w["end"])
                transcript = Transcript.from_words_info(
                    words=all_words,
                    start=(all_starts if all_starts else None),
                    end=(all_ends if all_ends else None),
                )
                # Clean up report files
                json_path.unlink(missing_ok=True)
                srt_path = engine_output.srt_report_path
                if srt_path:
                    srt_path.unlink(missing_ok=True)
            else:
                raise RuntimeError(f"Transcription report not found at {json_path}")

            logger.info("Transcription: " + transcript.get_transcript_string()[:100] + "...")

            return transcript

        return generate_and_transcribe

    def _build_transcription_engine(self) -> WhisperKitPro:
        """Create WhisperKitPro engine for transcription.

        Uses the same engine as the dedicated
        WhisperKitPro transcription pipelines, which
        handles model download, caching, and CLI args.
        """
        config = self.config
        cli_path = config.transcription_cli_path or config.cli_path

        import coremltools as ct

        engine_config = WhisperKitProConfig(
            repo_id=config.transcription_repo_id,
            model_variant=config.transcription_model_variant,
            model_dir=config.transcription_model_path,
            word_timestamps=config.transcription_word_timestamps,
            chunking_strategy=config.transcription_chunking_strategy,
            audio_encoder_compute_units=ct.ComputeUnit.CPU_AND_NE,
            text_decoder_compute_units=ct.ComputeUnit.CPU_AND_NE,
        )

        return WhisperKitPro(
            cli_path=cli_path,
            transcription_config=engine_config,
        )

    def __call__(self, input_sample: BaseSample) -> PipelineOutput:
        """Run pipeline and set generated audio duration.

        Overrides base __call__ to propagate the real
        TTS audio duration back onto the sample so the
        runner reports accurate audio_duration and
        speed_factor values.
        """
        self._last_generated_duration: float | None = None
        parsed_input = self.parse_input(input_sample)
        start_time = time.perf_counter()
        output = self.pipeline(parsed_input)
        end_time = time.perf_counter()
        prediction_time = end_time - start_time
        parsed_output = self.parse_output(output)
        if parsed_output.prediction_time is None:
            parsed_output.prediction_time = prediction_time

        # Propagate generated audio duration to sample
        dur = self._last_generated_duration
        logger.debug(f"Generated audio duration: {dur}s")
        if isinstance(input_sample, SpeechGenerationSample) and dur is not None:
            input_sample.generated_audio_duration = dur
            logger.debug(f"Set sample duration to {input_sample.generated_audio_duration}s")

        return parsed_output

    def parse_input(self, input_sample: SpeechGenerationSample) -> SpeechGenerationInput:
        """Extract text prompt from the sample."""
        text = input_sample.reference.get_transcript_string()
        return SpeechGenerationInput(
            text=text,
            audio_name=input_sample.audio_name,
        )

    def parse_output(self, output: Transcript) -> SpeechGenerationOutput:
        """Wrap transcription into output."""
        return SpeechGenerationOutput(prediction=output)
