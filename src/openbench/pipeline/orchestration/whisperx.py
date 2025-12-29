# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2025 Argmax, Inc. All Rights Reserved.

import subprocess
from pathlib import Path
from typing import Callable

import pandas as pd
from argmaxtools.utils import get_logger
from pydantic import Field

from ...dataset import DiarizationSample
from ...pipeline_prediction import Transcript
from ..base import Pipeline, PipelineType, register_pipeline
from .common import OrchestrationConfig, OrchestrationOutput


logger = get_logger(__name__)

TEMP_AUDIO_DIR = Path("audio_temp")


class WhisperXPipelineConfig(OrchestrationConfig):
    model_name: str = Field(
        default="tiny",
        description="The name of the Whisper model to use",
    )
    device: str = Field(
        default="cpu",
        description="The device to run the model on",
    )
    compute_type: str = Field(
        default="float32",
        description="The compute type to use for the model",
    )
    batch_size: int = Field(
        default=16,
        description="The batch size to use when transcribing the audio chunks",
    )
    threads: int = Field(
        default=0,
        description="Number of threads used by torch for CPU inference",
    )


class WhisperXInput:
    """Input for WhisperX CLI."""

    def __init__(self, audio_path: Path, language: str | None = None):
        self.audio_path = audio_path
        self.language = language


class WhisperX:
    def __init__(self, config: WhisperXPipelineConfig):
        self.config = config

    def __call__(self, input: WhisperXInput) -> pd.DataFrame:
        audio_path = input.audio_path
        if isinstance(audio_path, str):
            audio_path = Path(audio_path)

        audio_name = audio_path.stem
        # Convert config to CLI arguments
        output_dir = f"output/{audio_name}"
        args = [
            "whisperx",
            str(audio_path),
            "--output_dir",
            output_dir,
            "--model",
            self.config.model_name,
            "--device",
            self.config.device,
            "--batch_size",
            str(self.config.batch_size),
            "--compute_type",
            self.config.compute_type,
            "--threads",
            str(self.config.threads),
            "--diarize",
        ]

        # Add language if provided
        if input.language:
            args.extend(["--language", input.language])

        # Run whisperx CLI
        subprocess.run(args)

        # Parse .vtt file
        with open(f"{output_dir}/{audio_name}.vtt", "r") as f:
            vtt_content = f.read()

        # Parse VTT content
        lines = vtt_content.strip().split("\n")
        segments = []
        current_speaker = None

        for line in lines:
            line = line.strip()
            if not line or line == "WEBVTT":
                continue

            # Check if line contains timestamp
            if " --> " in line:
                start, end = line.split(" --> ")
                # Convert timestamp to seconds
                start_sec = start
                end_sec = end
            # Check if line contains speaker and text
            elif line.startswith("[SPEAKER_"):
                speaker_end = line.find("]:")
                if speaker_end != -1:
                    current_speaker = line[1:speaker_end]
                    text = line[speaker_end + 2 :].strip()
                    segments.append(
                        {
                            "start": start_sec,
                            "end": end_sec,
                            "speaker_label": current_speaker,
                            "text": text,
                        }
                    )
            # If line doesn't have speaker tag, use previous speaker
            elif line and current_speaker is not None:
                segments[-1]["text"] += " " + line

        # Convert to DataFrame
        df = pd.DataFrame(segments)

        # Clean up
        audio_path.unlink()

        return df


@register_pipeline
class WhisperXPipeline(Pipeline):
    _config_class = WhisperXPipelineConfig
    pipeline_type = PipelineType.ORCHESTRATION

    def build_pipeline(self) -> Callable[[WhisperXInput], pd.DataFrame]:
        return WhisperX(self.config)

    def parse_input(self, input_sample: DiarizationSample) -> WhisperXInput:
        # Extract language if force_language is enabled
        language = None
        if self.config.force_language:
            language = input_sample.extra_info.get("language", None)

        return WhisperXInput(
            audio_path=input_sample.save_audio(TEMP_AUDIO_DIR),
            language=language,
        )

    def parse_output(self, output: pd.DataFrame) -> OrchestrationOutput:
        output = output.assign(words=lambda df: df["text"].str.split()).explode("words")

        words = output["words"].tolist()
        speakers = output["speaker_label"].tolist()

        prediction = Transcript.from_words_info(
            words=words,
            speaker=speakers,
        )
        return OrchestrationOutput(
            prediction=prediction,
        )
