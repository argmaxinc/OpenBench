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

# Supported audio formats by AVAudioFile (Apple's Core Audio)
SUPPORTED_AUDIO_FORMATS = {".wav", ".aiff", ".aif", ".caf", ".m4a", ".mp3"}

SPEECH_ANALYZER_REPO_URL = "https://github.com/argmaxinc/apple-speechanalyzer-cli-example"
PRODUCT_NAME = "apple-speechanalyzer-cli"
TEMP_AUDIO_DIR = Path("./temp_audio")
SPEECH_ANALYZER_DEFAULT_REPORT_PATH = Path("speech_analyzer_report")
SPEECH_ANALYZER_DEFAULT_CLONE_DIR = "./speech_analyzer_repo"
# Commit hash for berkin/sfspeechrecognizer branch
SPEECH_ANALYZER_DEFAULT_COMMIT_HASH = "f47d46b83e79f5f687075de392f6689c758b67d7"


class SpeechAnalyzerConfig(TranscriptionConfig):
    clone_dir: str = Field(
        default=SPEECH_ANALYZER_DEFAULT_CLONE_DIR, description="The directory to clone the Speech Analyzer repo into"
    )
    commit_hash: str | None = Field(
        default=SPEECH_ANALYZER_DEFAULT_COMMIT_HASH,
        description="The commit hash of the Speech Analyzer repo when cloning (default: berkin/sfspeechrecognizer branch)",
    )
    use_server: bool = Field(
        default=False,
        description="Force server-based recognition (SFSpeechRecognizer) instead of on-device (SpeechTranscriber). "
        "Server mode is automatically enabled when use_keywords=True.",
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
        # --show-bin-path returns the directory, append product name to get full binary path
        bin_dir = result.stdout.strip()
        return f"{bin_dir}/{PRODUCT_NAME}"

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

    def _convert_to_wav(self, audio_path: Path) -> Path:
        """Convert audio to WAV format if not already supported by AVAudioFile."""
        if audio_path.suffix.lower() in SUPPORTED_AUDIO_FORMATS:
            return audio_path

        wav_path = audio_path.with_suffix(".wav")
        logger.debug(f"Converting {audio_path} to WAV format")
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(audio_path), "-ar", "16000", "-ac", "1", str(wav_path)],
                check=True,
                capture_output=True,
            )
        except FileNotFoundError:
            raise RuntimeError(
                "ffmpeg not found. Please install ffmpeg to convert audio formats. "
                "On macOS: brew install ffmpeg"
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to convert audio: {e.stderr.decode()}")

        # Remove original file
        audio_path.unlink(missing_ok=True)
        return wav_path

    def transcribe(self, input: SpeechAnalyzerCliInput, custom_phrases=None, use_server=False) -> Path:
        SPEECH_ANALYZER_DEFAULT_REPORT_PATH.mkdir(parents=True, exist_ok=True)

        # Convert audio to WAV if needed (AVAudioFile doesn't support FLAC)
        audio_path = self._convert_to_wav(input.audio_path)
        output_path = SPEECH_ANALYZER_DEFAULT_REPORT_PATH / audio_path.with_suffix(".txt").name

        # Use absolute paths since CLI runs with different cwd
        audio_path_abs = audio_path.resolve()
        output_path_abs = output_path.resolve()

        cmd = [
            self.cli_path,
            "--input-audio-path",
            str(audio_path_abs),
            "--output-text-path",
            str(output_path_abs),
        ]
        if input.language:
            cmd.extend(["--locale", input.language])

        # Force server-based recognition (SFSpeechRecognizer) if requested
        if use_server:
            cmd.append("--server")

        # Add custom phrases for vocabulary boosting
        # Reference: https://developer.apple.com/documentation/speech/analysiscontext
        if custom_phrases:
            cmd.extend(["--custom-phrases", ",".join(custom_phrases)])
            logger.debug(f"Using custom phrases: {custom_phrases}")

        subprocess.run(cmd, cwd=self.config.clone_dir, check=True)

        if not input.keep_audio:
            audio_path.unlink(missing_ok=True)

        return output_path


@register_pipeline
class SpeechAnalyzerPipeline(Pipeline):
    _config_class = SpeechAnalyzerConfig
    pipeline_type = PipelineType.TRANSCRIPTION

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_keywords = None

    def build_pipeline(self) -> Callable[[SpeechAnalyzerCliInput], Path]:
        engine = SpeechAnalyzerCli(config=self.config)

        def transcribe(input: SpeechAnalyzerCliInput) -> Path:
            response = engine.transcribe(
                input,
                custom_phrases=self.current_keywords,
                use_server=self.config.use_server,
            )
            return response

        return transcribe

    def parse_input(self, input_sample: TranscriptionSample) -> SpeechAnalyzerCliInput:
        """Override to extract keywords from sample before processing."""
        self.current_keywords = None
        if self.config.use_keywords:
            keywords = input_sample.extra_info.get("dictionary", [])
            if keywords:
                self.current_keywords = keywords

        return SpeechAnalyzerCliInput(
            audio_path=input_sample.save_audio(TEMP_AUDIO_DIR),
            keep_audio=False,
        )

    def parse_output(self, output: Path) -> TranscriptionOutput:
        transcription = output.read_text()
        return TranscriptionOutput(
            prediction=Transcript.from_words_info(words=transcription.split()),
        )
