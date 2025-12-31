# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2025 Argmax, Inc. All Rights Reserved.

import json
import os
import subprocess
from pathlib import Path
from typing import Callable

from argmaxtools.utils import _maybe_git_clone, get_logger
from pydantic import BaseModel, Field

from ...dataset import TranscriptionSample
from ...pipeline_prediction import Transcript
from ..base import Pipeline, PipelineType, register_pipeline
from .common import TranscriptionConfig, TranscriptionOutput


logger = get_logger(__name__)

# Constants
WHISPERKIT_REPO_URL = "https://github.com/argmaxinc/WhisperKit"
PRODUCT_NAME = "whisperkit-cli"
TEMP_AUDIO_DIR = Path("./temp_audio")
WHISPERKIT_DEFAULT_REPORT_PATH = "./whisperkit_report"


class WhisperKitTranscriptionConfig(TranscriptionConfig):
    """Configuration for WhisperKit transcription operations."""

    model_version: str = Field(
        default="base",
        description="The version of the WhisperKit model to use (e.g., 'tiny', 'base', 'small', 'large-v3')",
    )
    word_timestamps: bool = Field(
        default=True,
        description="Whether to include word timestamps in the output",
    )
    chunking_strategy: str | None = Field(
        default="vad",
        description="The chunking strategy to use either `none` or `vad`",
    )
    report_path: str | None = Field(
        default=WHISPERKIT_DEFAULT_REPORT_PATH,
        description="The path to the directory where the report files will be saved. If not provided, the report files will be saved in the current working directory.",
    )
    prompt: str | None = Field(
        default=None,
        description="Initial prompt for transcription",
    )
    text_decoder_compute_units: str = Field(
        default="cpuAndNeuralEngine",
        description="Compute units for text decoder",
    )
    audio_encoder_compute_units: str = Field(
        default="cpuAndNeuralEngine",
        description="Compute units for audio encoder",
    )

    def create_report_path(self) -> Path:
        if self.report_path is None:
            return Path.cwd()

        report_dir = Path(self.report_path)

        if report_dir.exists():
            logger.info(f"Report dir already exists for WhisperKit at: {report_dir}")
            return report_dir

        report_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created report dir for WhisperKit at: {report_dir}")
        return report_dir

    def generate_cli_args(self) -> list[str]:
        args = [
            "--model",
            self.model_version,
            "--report",  # Always generate the report files
        ]

        if self.chunking_strategy:
            args.extend(["--chunking-strategy", self.chunking_strategy])
        if self.word_timestamps:
            args.append("--word-timestamps")
        if self.report_path:
            args.extend(["--report-path", self.report_path])
        if self.prompt:
            args.extend(["--prompt", f'"{self.prompt}"'])
        if self.text_decoder_compute_units:
            args.extend(["--text-decoder-compute-units", self.text_decoder_compute_units])
        if self.audio_encoder_compute_units:
            args.extend(["--audio-encoder-compute-units", self.audio_encoder_compute_units])

        logger.info(f"Generating CLI args for WhisperKit: {args}")
        return args


class WhisperKitEngineConfig(BaseModel):
    """Base configuration for WhisperKit operations."""

    commit_hash: str | None = Field(
        default=None,
        description="The commit hash of the WhisperKit repo when cloning",
    )
    cli_path: str | None = Field(
        default=None,
        description="The path to the WhisperKit CLI",
    )
    clone_dir: str = Field(
        default="./whisperkit_repo",
        description="Directory to clone and build the CLI",
    )


class TranscriptionCliInput(BaseModel):
    """Input for transcription CLI."""

    audio_path: Path
    keep_audio: bool = False
    language: str | None = None


class TranscriptionCliOutput(BaseModel):
    """Output for transcription CLI."""

    json_report_path: Path = Field(
        ...,
        description="Path to the JSON report with transcription results",
    )
    srt_report_path: Path = Field(
        ...,
        description="Path to the .srt file containing transcription results",
    )


class WhisperKitEngine:
    """Unified CLI interface for WhisperKit operations."""

    def __init__(
        self,
        config: WhisperKitEngineConfig,
        transcription_config: WhisperKitTranscriptionConfig,
    ):
        self.config = config
        self.cli_path = config.cli_path or self._clone_and_build_cli()
        self.transcription_config = transcription_config
        self.transcription_args = self.transcription_config.generate_cli_args()
        self.transcription_config.create_report_path()

    def _clone_and_build_cli(self) -> str:
        """Clone the repository and build the CLI."""
        os.makedirs(self.config.clone_dir, exist_ok=True)
        if not WHISPERKIT_REPO_URL:
            raise ValueError("Repository URL is not set")

        logger.info(f"Cloning repo {WHISPERKIT_REPO_URL} into {self.config.clone_dir}")
        repo_name = WHISPERKIT_REPO_URL.split("/")[-1]
        repo_owner = WHISPERKIT_REPO_URL.split("/")[-2]

        repo_dir, commit_hash = _maybe_git_clone(
            out_dir=self.config.clone_dir,
            hub_url="github.com",
            repo_name=repo_name,
            repo_owner=repo_owner,
            commit_hash=self.config.commit_hash,
        )
        logger.info(f"{repo_name} -> Commit hash: {commit_hash}")

        try:
            build_dir = self._build_cli(repo_dir)
            cli_path = os.path.join(build_dir, PRODUCT_NAME)
            self.config.commit_hash = commit_hash
            return cli_path
        except subprocess.CalledProcessError as e:
            logger.error(f"Build failed with return code {e.returncode}")
            logger.error(f"Build stdout:\n{e.stdout}")
            logger.error(f"Build stderr:\n{e.stderr}")
            raise RuntimeError(
                f"Failed to build CLI: Exit code {e.returncode}\nStdout: {e.stdout}\nStderr: {e.stderr}"
            )

    def _build_cli(self, repo_dir: str) -> str:
        """Build the CLI and return the build directory path."""
        logger.info(f"Building {PRODUCT_NAME} CLI...")

        build_cmd = f"swift build -c release --product {PRODUCT_NAME}"

        subprocess.run(
            build_cmd,
            cwd=repo_dir,
            shell=True,
            check=True,
        )
        logger.info(f"Successfully built {PRODUCT_NAME} CLI!")

        result = subprocess.run(
            f"{build_cmd} --show-bin-path",
            cwd=repo_dir,
            stdout=subprocess.PIPE,
            shell=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()

    def transcribe(self, input: TranscriptionCliInput) -> TranscriptionCliOutput:
        """Run transcription on the given audio file."""
        cmd = [
            self.cli_path,
            "transcribe",
            "--audio-path",
            str(input.audio_path),
            *self.transcription_args,
        ]
        if input.language:
            cmd.extend(["--language", input.language])

        logger.debug(f"Running WhisperKit CLI: {cmd}")

        report_dir = self.transcription_config.create_report_path()
        if not report_dir:
            raise ValueError("Report directory not configured")

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"CLI command failed: {e.stderr}")

        if not input.keep_audio:
            input.audio_path.unlink(missing_ok=True)

        json_report_path = report_dir / input.audio_path.with_suffix(".json").name
        srt_report_path = report_dir / input.audio_path.with_suffix(".srt").name

        return TranscriptionCliOutput(
            json_report_path=json_report_path,
            srt_report_path=srt_report_path,
        )


@register_pipeline
class WhisperKitTranscriptionPipeline(Pipeline):
    _config_class = WhisperKitTranscriptionConfig
    pipeline_type = PipelineType.TRANSCRIPTION

    def build_pipeline(self) -> Callable[[TranscriptionCliInput], TranscriptionCliOutput]:
        # Create WhisperKit engine
        engine_config = WhisperKitEngineConfig(
            clone_dir="./whisperkit_repo",
        )

        engine = WhisperKitEngine(
            config=engine_config,
            transcription_config=self.config,
        )

        return engine.transcribe

    def parse_input(self, input_sample: TranscriptionSample) -> TranscriptionCliInput:
        # Extract language if force_language is enabled
        language = None
        if self.config.force_language:
            language = input_sample.language

        return TranscriptionCliInput(
            audio_path=input_sample.save_audio(TEMP_AUDIO_DIR),
            keep_audio=False,
            language=language,
        )

    def parse_output(self, output: TranscriptionCliOutput) -> TranscriptionOutput:
        """Parse JSON output file into TranscriptionOutput."""
        with output.json_report_path.open("r") as f:
            data = json.load(f)

        transcript = Transcript.from_words_info(
            words=[word["word"] for segment in data["segments"] for word in segment["words"]],
            start=[word["start"] for segment in data["segments"] for word in segment["words"] if "start" in word],
            end=[word["end"] for segment in data["segments"] for word in segment["words"] if "end" in word],
        )

        return TranscriptionOutput(
            prediction=transcript,
        )
