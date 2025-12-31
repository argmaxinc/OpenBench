# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2025 Argmax, Inc. All Rights Reserved.

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

SPEECH_ANALYZER_REPO_URL = "https://github.com/argmaxinc/apple-speechanalyzer-cli-example"
PRODUCT_NAME = "apple-speechanalyzer-cli"
TEMP_AUDIO_DIR = Path("./temp_audio")
SPEECH_ANALYZER_DEFAULT_REPORT_PATH = Path("speech_analyzer_report")
SPEECH_ANALYZER_DEFAULT_CLONE_DIR = "./speech_analyzer_repo"


class SpeechAnalyzerConfig(TranscriptionConfig):
    clone_dir: str = Field(
        default=SPEECH_ANALYZER_DEFAULT_CLONE_DIR, description="The directory to clone the Speech Analyzer repo into"
    )
    commit_hash: str | None = Field(
        default=None, description="The commit hash of the Speech Analyzer repo when cloning"
    )


class SpeechAnalyzerCliInput(BaseModel):
    audio_path: Path
    keep_audio: bool = False
    language: str | None = None


class SpeechAnalyzerCli:
    def __init__(self, config: SpeechAnalyzerConfig) -> None:
        self.config = config
        self.cli_path = self._clone_and_build_cli()

    def _build_cli(self, repo_dir: str) -> str:
        build_cmd = f"swift build -c release --product {PRODUCT_NAME}"
        try:
            subprocess.run(build_cmd, cwd=repo_dir, check=True, shell=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Failed to build Speech Analyzer CLI with command: {build_cmd}\n"
                f"Exit code: {e.returncode}\n"
                f"Output: {e.output}\n"
                f"Stdout: {getattr(e, 'stdout', None)}\n"
                f"Stderr: {getattr(e, 'stderr', None)}"
            ) from e

        # Get the path to the built CLI
        try:
            result = subprocess.run(
                f"{build_cmd} --show-bin-path",
                cwd=repo_dir,
                stdout=subprocess.PIPE,
                shell=True,
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Failed to get Speech Analyzer CLI binary path with command: {build_cmd} --show-bin-path\n"
                f"Exit code: {e.returncode}\n"
                f"Output: {e.output}\n"
                f"Stdout: {getattr(e, 'stdout', None)}\n"
                f"Stderr: {getattr(e, 'stderr', None)}"
            ) from e
        return result.stdout.strip()

    def _clone_and_build_cli(self) -> None:
        repo_name = SPEECH_ANALYZER_REPO_URL.split("/")[-1]
        repo_owner = SPEECH_ANALYZER_REPO_URL.split("/")[-2]
        repo_dir, commit_hash = _maybe_git_clone(
            out_dir=self.config.clone_dir,
            hub_url="github.com",
            repo_name=repo_name,
            repo_owner=repo_owner,
            commit_hash=self.config.commit_hash,
        )
        self.config.commit_hash = commit_hash

        return self._build_cli(repo_dir)

    def transcribe(self, input: SpeechAnalyzerCliInput) -> Path:
        SPEECH_ANALYZER_DEFAULT_REPORT_PATH.mkdir(parents=True, exist_ok=True)
        output_path = SPEECH_ANALYZER_DEFAULT_REPORT_PATH / input.audio_path.with_suffix(".txt").name

        cmd = [
            self.cli_path,
            "--input-audio-path",
            str(input.audio_path),
            "--output-text-path",
            str(output_path),
        ]
        if input.language:
            cmd.extend(["--locale", input.language])

        subprocess.run(cmd, cwd=self.config.clone_dir, check=True)

        if not input.keep_audio:
            input.audio_path.unlink(missing_ok=True)

        return output_path


@register_pipeline
class SpeechAnalyzerPipeline(Pipeline):
    _config_class = SpeechAnalyzerConfig
    pipeline_type = PipelineType.TRANSCRIPTION

    def build_pipeline(self) -> Callable[[SpeechAnalyzerCliInput], Path]:
        engine = SpeechAnalyzerCli(config=self.config)
        return engine.transcribe

    def parse_input(self, input_sample: TranscriptionSample) -> SpeechAnalyzerCliInput:
        # Extract language if force_language is enabled
        language = None
        if self.config.force_language:
            language = input_sample.language

        return SpeechAnalyzerCliInput(
            audio_path=input_sample.save_audio(TEMP_AUDIO_DIR),
            keep_audio=False,
            language=language,
        )

    def parse_output(self, output: Path) -> TranscriptionOutput:
        transcription = output.read_text()
        return TranscriptionOutput(
            prediction=Transcript.from_words_info(words=transcription.split()),
        )
